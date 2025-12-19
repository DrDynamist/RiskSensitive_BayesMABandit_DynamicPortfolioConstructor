# src/config.py

ASSETS = ["SPY", "AAPL", "MSFT", "NVDA", "TLT", "GLD"]

START_DATE = "2010-01-01"
END_DATE = "2025-11-30"
PRESAMPLE_END = "2014-12-31"
BANDIT_START = "2015-01-01"

# Risk parameters (for later)
ALPHA_CVAR = 0.05
LAMBDA_RISK = 3.0

# Normal–Inverse-Gamma prior hyperparameters
# μ ~ Normal(m0, σ² / κ0)
# σ² ~ Inverse-Gamma(α0, β0)
M0 = 0.0
KAPPA0 = 0.01
ALPHA0 = 2.0
BETA0 = 2e-4
