
import os, random
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def calc_standard_f1(y_score, y_true, steps=600):
    best_std_f1, best_std_p, best_std_r = 0.0, 0.0, 0.0

    th_quantiles = np.linspace(0.80, 0.999, steps // 2)
    ths_q = np.quantile(y_score, th_quantiles)
    ths_l = np.linspace(y_score.min(), y_score.max(), steps // 2)
    thresholds = np.unique(np.concatenate([ths_q, ths_l]))

    for th in thresholds:
        pred = (y_score >= th).astype(np.int32)
        p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average='binary', zero_division=0)
        if f1 > best_std_f1:
            best_std_f1, best_std_p, best_std_r = f1, p, r

    return float(best_std_p), float(best_std_r), float(best_std_f1)