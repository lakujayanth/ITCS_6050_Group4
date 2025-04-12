# Default configuration
DEFAULT_CONFIG = {
    "num_iterations": 50,
    "k_shots": 5,
    "meta_lr": 0.001,
    "inner_lr": 0.01,
    "hidden_dims": [128, 64],
    "conv_filters": [16, 32],
    "num_episodes_eval": 3,
    "log_interval": 5,
}

# Multiple configurations for batch tuning
TUNING_CONFIGS = [
    DEFAULT_CONFIG,
    {
        "num_iterations": 100,
        "k_shots": 10,
        "meta_lr": 0.0005,
        "inner_lr": 0.02,
        "hidden_dims": [256, 128],
        "conv_filters": [32, 64],
        "num_episodes_eval": 5,
        "log_interval": 2,
    },
    {
        "num_iterations": 50,
        "k_shots": 5,
        "meta_lr": 0.0001,
        "inner_lr": 0.05,
        "hidden_dims": [128, 64],
        "conv_filters": [16, 32],
        "num_episodes_eval": 3,
        "log_interval": 5,
    },
]