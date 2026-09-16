import os
import sys
import random
import warnings
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import MinMaxScaler

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.ultimate_net import UltimateDinoTFMAE
from utils.config import SIGNAL_PHYSICAL_CONFIGS
from utils.profiler import DataCharacteristicProfiler
from utils.data_loader import load_and_preprocess_dataset
from utils.postprocess import calibrate_score, apply_post_processing
from vus.metrics_helper import seed_everything, calc_standard_f1
from vus.metrics import get_metrics

WINDOW_SIZE = 224
STRIDE = 16


def run_trial(base_seed, all_data_tensor, test_data, true_labels_raw, CHANNELS, device, cfg, weights):
    seed_everything(base_seed)
    BATCH_SIZE, ACCUM_STEPS = 4, 8

    train_loader = [
        all_data_tensor[b * BATCH_SIZE: (b + 1) * BATCH_SIZE] 
        for b in range(len(all_data_tensor) // BATCH_SIZE)
    ]

    ultimate_net = UltimateDinoTFMAE(ts_channels=CHANNELS, ts_length=WINDOW_SIZE, device=device).to(device)

    opt_spatial_dino = torch.optim.AdamW([
        {"params": ultimate_net.spatial_mae.parameters()},
        {"params": ultimate_net.dino_to_ts.parameters()},
    ], lr=4e-4, weight_decay=0.04)
    opt_temporal = torch.optim.AdamW(ultimate_net.temporal_mae.parameters(), lr=4e-4, weight_decay=0.03)

    cached_train_dino_feats = []
    ultimate_net.eval()
    with torch.no_grad(), autocast():
        for chunk in all_data_tensor:
            chunk_dev = chunk.to(device, non_blocking=True)
            rgb_tensor = ultimate_net.project_to_dinov2_space(chunk_dev.unsqueeze(0))
            feats = ultimate_net.dinov2_extractor(rgb_tensor)
            cached_train_dino_feats.append(feats.squeeze(0).detach())
    cached_train_dino_feats = torch.stack(cached_train_dino_feats)

    epochs = 10
    sched_spatial = torch.optim.lr_scheduler.CosineAnnealingLR(opt_spatial_dino, T_max=epochs, eta_min=1e-6)
    sched_temporal = torch.optim.lr_scheduler.CosineAnnealingLR(opt_temporal, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    for epoch in range(epochs):
        ultimate_net.train()
        train_indices = list(range(len(train_loader)))
        random.shuffle(train_indices)
        opt_spatial_dino.zero_grad()
        opt_temporal.zero_grad()
        adv_warmup = min(1.0, (epoch + 1) / 3.0)

        for step, m_idx in enumerate(train_indices):
            x_batch = train_loader[m_idx].to(device, non_blocking=True)
            start_b, end_b = m_idx * BATCH_SIZE, (m_idx + 1) * BATCH_SIZE

            with autocast():
                P_out, loss_t = ultimate_net.temporal_mae(x_batch, mask_ratio=0.75, is_training=True)
                s_purified, loss_s, _ = ultimate_net.spatial_mae(x_batch, mask_ratio=0.40, is_training=True)
                dino_feats = cached_train_dino_feats[start_b:end_b].to(device, non_blocking=True)
                mapped = ultimate_net.dino_to_ts(dino_feats)
                Z_dino = mapped.view(mapped.shape[0], CHANNELS, WINDOW_SIZE)

                raw_contrast = ultimate_net.compute_symmetric_kld(P_out.detach(), Z_dino, 1.0) * adv_warmup
                loss_phase1 = (torch.clamp(raw_contrast, -10.0, 10.0) * 0.1 + cfg["spatial_loss_weight"] * loss_s + loss_t) / ACCUM_STEPS
            scaler.scale(loss_phase1).backward()

            with autocast():
                P_out_adv, _ = ultimate_net.temporal_mae(x_batch, mask_ratio=0.75, is_training=True)
                raw_adv = -ultimate_net.compute_symmetric_kld(P_out_adv, Z_dino.detach(), 1.0) * adv_warmup
                loss_adv = torch.clamp(raw_adv, -10.0, 10.0) * 0.1 / ACCUM_STEPS
            scaler.scale(loss_adv).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(opt_spatial_dino)
                scaler.unscale_(opt_temporal)
                torch.nn.utils.clip_grad_norm_(ultimate_net.parameters(), max_norm=5.0)
                scaler.step(opt_spatial_dino)
                scaler.step(opt_temporal)
                scaler.update()
                opt_spatial_dino.zero_grad()
                opt_temporal.zero_grad()

        sched_spatial.step()
        sched_temporal.step()

    total_len = len(test_data)
    global_kld = np.zeros(total_len, dtype=np.float32)
    global_spatial = np.zeros(total_len, dtype=np.float32)
    global_temporal = np.zeros(total_len, dtype=np.float32)
    global_counts = np.zeros(total_len, dtype=np.float32)

    ultimate_net.eval()
    k_core = min(cfg["top_k_core"], CHANNELS)
    k_context = min(cfg["top_k_context"], CHANNELS)
    alpha_pool = cfg["pooling_ratio"]
    w_vis_norm, w_sp_norm, w_tp_norm = weights

    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    start_indices = list(range(0, total_len - WINDOW_SIZE + 1, STRIDE))

    with torch.no_grad(), autocast():
        for b_start in range(0, len(start_indices), 128):
            b_starts = start_indices[b_start: b_start + 128]
            seg_batch = torch.stack([test_tensor[s:s + WINDOW_SIZE].T for s in b_starts])

            P_pure, _ = ultimate_net.temporal_mae(seg_batch, mask_ratio=0.0, is_training=False)
            S_pure, _, _ = ultimate_net.spatial_mae(seg_batch, mask_ratio=0.0, is_training=False)

            top_vals_sp = torch.topk((S_pure - seg_batch) ** 2, k=k_context, dim=1)[0]
            l2_spatial = (alpha_pool * top_vals_sp[:, :k_core].mean(dim=1) + (1.0 - alpha_pool) * top_vals_sp.mean(dim=1)).cpu().numpy()

            top_vals_tp = torch.topk((P_pure - seg_batch) ** 2, k=k_context, dim=1)[0]
            l2_temporal = (alpha_pool * top_vals_tp[:, :k_core].mean(dim=1) + (1.0 - alpha_pool) * top_vals_tp.mean(dim=1)).cpu().numpy()

            if w_vis_norm > 0.0:
                dino_feats_raw = ultimate_net.dinov2_extractor(ultimate_net.project_to_dinov2_space(seg_batch))
                mapped = ultimate_net.dino_to_ts(dino_feats_raw)
                Z_dino_raw = mapped.view(mapped.shape[0], CHANNELS, WINDOW_SIZE)
                p_p = torch.softmax(torch.clamp(P_pure.float(), -20.0, 20.0), dim=-1)
                f_p = torch.softmax(torch.clamp(Z_dino_raw.float(), -20.0, 20.0), dim=-1)
                kld_matrix = p_p * (torch.log(p_p + 1e-5) - torch.log(f_p + 1e-5))
                kld_score = torch.sum(kld_matrix, dim=-2).cpu().numpy()
            else:
                kld_score = np.zeros((len(b_starts), WINDOW_SIZE), dtype=np.float32)

            damping = cfg["damping_scale"]
            if damping > 0.0:
                seg_diff = torch.abs(seg_batch[:, :, 1:] - seg_batch[:, :, :-1])
                gate_m = torch.max(F.pad(seg_diff, (1, 0), mode='replicate'), dim=1)[0]
                w_gate = (1.0 / (1.0 + damping * gate_m)).cpu().numpy()
                kld_score *= w_gate
                l2_spatial *= w_gate
                l2_temporal *= w_gate

            idx_mat = (np.array(b_starts)[:, None] + np.arange(WINDOW_SIZE)).ravel()
            np.add.at(global_kld, idx_mat, kld_score.ravel())
            np.add.at(global_spatial, idx_mat, l2_spatial.ravel())
            np.add.at(global_temporal, idx_mat, l2_temporal.ravel())
            np.add.at(global_counts, idx_mat, 1.0)

    max_idx = ((total_len - WINDOW_SIZE) // STRIDE) * STRIDE + WINDOW_SIZE
    counts = np.maximum(global_counts[:max_idx], 1e-8)
    s_spatial = global_spatial[:max_idx] / counts
    s_temporal = global_temporal[:max_idx] / counts

    c_w = cfg["center_weight"]
    norm_spatial = calibrate_score(s_spatial, center_weight=c_w)
    norm_temporal = calibrate_score(s_temporal, center_weight=c_w)
    norm_kld = calibrate_score(global_kld[:max_idx] / counts, center_weight=c_w) if w_vis_norm > 0.0 else 0.0

    r = cfg["rectify_ratio"]
    spatial_term = r * np.maximum(0.0, norm_spatial) + (1.0 - r) * norm_spatial
    temporal_term = r * np.maximum(0.0, norm_temporal) + (1.0 - r) * norm_temporal

    y_score_raw = w_vis_norm * norm_kld + w_sp_norm * spatial_term + w_tp_norm * temporal_term
    q = cfg["baseline_quantile"]
    base_line = (1.0 - r) * (np.percentile(y_score_raw, q) if q > 0.0 else np.min(y_score_raw))

    y_score_raw = np.log1p(np.maximum(0.0, y_score_raw - base_line) + 1e-5)
    y_score = apply_post_processing(y_score_raw, cfg)
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=1.0, neginf=0.0)
    rng = y_score.max() - y_score.min()
    y_score_norm = np.zeros_like(y_score) if rng < 1e-8 else MinMaxScaler((0, 1)).fit_transform(y_score.reshape(-1, 1)).ravel()

    y_true = (np.array(true_labels_raw[:max_idx]).flatten() > 0).astype(np.int32)
    _, _, std_f1 = calc_standard_f1(y_score_norm, y_true, steps=600)
    vus_res = get_metrics(y_score_norm, y_true, metric='vus', slidingWindow=WINDOW_SIZE)

    return {
        'std_f1': std_f1,
        'V_ROC': vus_res.get('VUS_ROC', 0.0) * 100.0,
        'V_PR':  vus_res.get('VUS_PR',  0.0) * 100.0,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ViST-MAE Autonomous Continuous Physical Profiler & Engine")
    parser.add_argument("--dataset", type=str, default="SMD", choices=["SWaT", "SMAP", "MSL", "SMD", "PSM"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_data_raw, test_data, true_labels_raw, all_data_tensor, CHANNELS = load_and_preprocess_dataset(
        dataset_name=args.dataset, project_dir=PROJECT_DIR, window_size=WINDOW_SIZE
    )

    profiler = DataCharacteristicProfiler()
    profile = profiler.profile_and_route(train_data_raw)
    gamma, beta, alpha = profile["convex_weights"]

    cfg = SIGNAL_PHYSICAL_CONFIGS.get(args.dataset, SIGNAL_PHYSICAL_CONFIGS["PSM"]).copy()
    print(f"Running seeds: {args.seeds} ...")

    all_reports = []
    for s in args.seeds:
        res = run_trial(s, all_data_tensor, test_data, true_labels_raw, CHANNELS, device, cfg, (gamma, beta, alpha))
        all_reports.append(res)
        print(f"Seed {s:2d} -> VUS-ROC: {res['V_ROC']:5.2f}% | VUS-PR: {res['V_PR']:5.2f}% | F1: {res['std_f1']:.4f}")

    print("------------------------------------------------------------")
    print(f"【{args.dataset} Final (Mean ± Std over {len(args.seeds)} seeds)】")
    print(f"VUS-ROC : {np.mean([r['V_ROC'] for r in all_reports]):5.2f}% ± {np.std([r['V_ROC'] for r in all_reports]):.2f}%")
    print(f"VUS-PR  : {np.mean([r['V_PR'] for r in all_reports]):5.2f}% ± {np.std([r['V_PR'] for r in all_reports]):.2f}%")
    print(f"Best-F1 : {np.mean([r['std_f1'] for r in all_reports]):.4f} ± {np.std([r['std_f1'] for r in all_reports]):.4f}")
    print("------------------------------------------------------------")