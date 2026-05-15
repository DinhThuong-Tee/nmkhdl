import pandas as pd
import numpy as np

# =========================
# SYNTHETIC FUNCTIONS
# =========================
def add_synthetic_h2s(df, mean=0.04, std=0.015, seed=42):
    rng = np.random.default_rng(seed)
    mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))

    h2s = rng.lognormal(mu, sigma, size=len(df))
    h2s = np.clip(h2s, 0.0005, 0.06)

    df["H2S"] = h2s
    return df

