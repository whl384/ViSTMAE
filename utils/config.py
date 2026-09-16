SIGNAL_PHYSICAL_CONFIGS = {
    "SWaT": {
        "spatial_loss_weight": 1.1, "center_weight": 1.0, "rectify_ratio": 1.0, 
        "baseline_quantile": 0.06, "damping_scale": 0.0, "top_k_core": 8, "top_k_context": 12, 
        "pooling_ratio": 0.95,
        "sigma": 3.4, "rise": 0.25, "decay": 0.15,
        "sigma_aux": 0.0, "rise_aux": 0.0, "decay_aux": 0.0,
    },
    "SMD": {
        "spatial_loss_weight": 1.0, "center_weight": 0.0, "rectify_ratio": 0.0, 
        "baseline_quantile": 0.08, "damping_scale": 0.0, "top_k_core": 6, "top_k_context": 14, 
        "pooling_ratio": 0.70,
        "sigma": 2.8, "rise": 0.24, "decay": 0.026,
        "sigma_aux": 0.0, "rise_aux": 0.0, "decay_aux": 0.0,
    },
    "MSL": {
        "spatial_loss_weight": 1.0, "center_weight": 0.0, "rectify_ratio": 0.45, 
        "baseline_quantile": 0.0, "damping_scale": 0.0, "top_k_core": 16, "top_k_context": 16, 
        "pooling_ratio": 0.95,
        "sigma": 4.6, "rise": 0.24, "decay": 0.046,
        "sigma_aux": 10.2, "rise_aux": 0.11, "decay_aux": 0.014,
    },
    "SMAP": {
        "spatial_loss_weight": 1.0, "center_weight": 1.0, "rectify_ratio": 0.0, 
        "baseline_quantile": 0.04, "damping_scale": 45.0, "top_k_core": 16, "top_k_context": 16, 
        "pooling_ratio": 0.95,
        "sigma": 6.6, "rise": 0.14, "decay": 0.045,
        "sigma_aux": 9.2, "rise_aux": 0.05, "decay_aux": 0.020,
    },
    "PSM": {
        "spatial_loss_weight": 1.0, "center_weight": 0.0, "rectify_ratio": 0.0, 
        "baseline_quantile": 0.045, "damping_scale": 0.0, "top_k_core": 5, "top_k_context": 10, 
        "pooling_ratio": 0.95,
        "sigma": 3.0, "rise": 0.26, "decay": 0.055,
        "sigma_aux": 0.0, "rise_aux": 0.0, "decay_aux": 0.0,
    },
}