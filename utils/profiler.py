import numpy as np


class DataCharacteristicProfiler:
    def __init__(self, tau_s: float = 0.25, tau_t: float = 0.60, tau_m: float = 0.30):
        self.tau_s = tau_s  
        self.tau_t = tau_t  
        self.tau_m = tau_m  

    def profile_and_route(self, train_data: np.ndarray) -> dict:
        T, C = train_data.shape
        data = np.nan_to_num(train_data, nan=0.0)

        if C > 1:
            corr_mat = np.nan_to_num(np.corrcoef(data, rowvar=False), nan=0.0)
            upper_tri = np.abs(corr_mat[np.triu_indices(C, k=1)])
            s_corr = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
        else:
            s_corr = 0.0

        t_corrs = []
        for c in range(C):
            col = data[:, c]
            if np.std(col) > 1e-4:
                r = np.corrcoef(col[:-1], col[1:])[0, 1]
                t_corrs.append(np.abs(r) if not np.isnan(r) else 0.0)
        t_auto = float(np.mean(t_corrs)) if len(t_corrs) > 0 else 0.5

        disc_channels, actuator_channels = 0, 0
        for c in range(C):
            n_unique = len(np.unique(data[:, c]))
            if n_unique <= 15:
                disc_channels += 1
            if n_unique <= 3:
                actuator_channels += 1

        m_disc = float(disc_channels / C)
        actuator_ratio = float(actuator_channels / C)

        if s_corr >= self.tau_s:
            r_spatial = (s_corr - self.tau_s + 0.10) * (1.0 + 2.0 * actuator_ratio)
        elif C >= 30 and t_auto >= self.tau_t:
            r_spatial = 0.12 * (s_corr / self.tau_s)
        else:
            r_spatial = 0.0

        if s_corr < self.tau_s and m_disc >= self.tau_m:
            r_vision = (m_disc - self.tau_m + 0.50) * np.exp(-s_corr / max(1e-4, self.tau_s))
        elif C >= 30 and t_auto >= self.tau_t and s_corr < self.tau_s:
            r_vision = 0.70 * t_auto
        else:
            r_vision = 0.0

        if s_corr >= self.tau_s:
            r_temporal = t_auto * (0.22 if actuator_ratio >= 0.20 else 0.75)
        elif t_auto >= self.tau_t:
            r_temporal = t_auto
        else:
            r_temporal = 0.50

        r_sum = r_vision + r_spatial + r_temporal
        if r_sum > 1e-6:
            gamma = float(r_vision / r_sum)
            beta  = float(r_spatial / r_sum)
            alpha = float(r_temporal / r_sum)
        else:
            alpha, beta, gamma = 1.0, 0.0, 0.0

        return {
            "S_corr": s_corr,
            "T_auto": t_auto,
            "M_disc": m_disc,
            "actuator_ratio": actuator_ratio,
            "convex_weights": (gamma, beta, alpha),
        }