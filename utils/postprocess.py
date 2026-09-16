import numpy as np
import torch

def pure_gaussian_smoothing(scores, sigma=3.0):
    radius = int(4.0 * sigma + 0.5)
    if radius == 0: 
        return scores
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 / (sigma ** 2) * (x ** 2))
    return np.convolve(np.pad(scores, (radius, radius), mode='edge'), k / k.sum(), mode='valid')


def safe_symmetric_kld(self, p, f, temp=1.0):
    p_fp32 = p.float()
    f_fp32 = f.float()
    if isinstance(temp, torch.Tensor): 
        temp = temp.mean().clamp(min=0.1)
    p_clamped = torch.clamp(p_fp32 / temp, min=-20.0, max=20.0)
    f_clamped = torch.clamp(f_fp32 / temp, min=-20.0, max=20.0)
    p_prob, f_prob = torch.softmax(p_clamped, dim=-1), torch.softmax(f_clamped, dim=-1)
    kld_p_f = torch.sum(p_prob * (torch.log(p_prob + 1e-5) - torch.log(f_prob + 1e-5)), dim=-1)
    kld_f_p = torch.sum(f_prob * (torch.log(f_prob + 1e-5) - torch.log(p_prob + 1e-5)), dim=-1)
    return torch.nan_to_num((kld_p_f + kld_f_p).mean(), nan=0.0, posinf=10.0, neginf=-10.0)


def calibrate_score(s, center_weight=0.0):
    s = np.nan_to_num(s, nan=0.0)
    if center_weight > 0.5:
        med = np.median(s)
        mad = np.median(np.abs(s - med))
        if mad < 1e-6:
            std_val = np.std(s)
            return (s - np.mean(s)) / (std_val if std_val >= 1e-8 else 1.0)
        s_centered = s - med
        s_robust = np.sign(s_centered) * np.log1p(np.abs(s_centered) / (mad + 1e-5))
        std_val = np.std(s_robust)
        return (s_robust - np.mean(s_robust)) / (std_val if std_val >= 1e-8 else 1.0)
    else:
        s_robust = np.log1p(np.clip(s, a_min=0, a_max=None))
        std_val = np.std(s_robust)
        return (s_robust - np.mean(s_robust)) / (std_val if std_val >= 1e-8 else 1.0)


def apply_post_processing(y_score_raw, cfg):
    def asymmetric_filter(raw, sig, r, d):
        s_sm = pure_gaussian_smoothing(raw, sigma=sig)
        out = np.zeros_like(s_sm)
        st = 0.0
        for t in range(len(s_sm)):
            st = (1 - r) * st + r * s_sm[t] if s_sm[t] >= st else (1 - d) * st + d * s_sm[t]
            out[t] = st
        return out

    out_primary = asymmetric_filter(y_score_raw, cfg["sigma"], cfg["rise"], cfg["decay"])
    
    if cfg["sigma_aux"] > 0.0:
        out_aux = asymmetric_filter(y_score_raw, cfg["sigma_aux"], cfg["rise_aux"], cfg["decay_aux"])
        return np.maximum(out_primary, out_aux)
    
    return out_primary