# src/policies.py

from typing import Dict
from .bayes_model import BayesianArm
from .config import LAMBDA_RISK, ALPHA_CVAR


def thompson_sampling_mean(arms: Dict[str, BayesianArm]) -> str:
    """
    Risk-neutral Thompson Sampling:
    - sample one predictive return from each arm
    - pick the arm with the highest sampled return
    """
    samples = {name: arm.sample_predictive(size=1)[0] for name, arm in arms.items()}
    return max(samples.keys(), key=lambda k: samples[k])


def thompson_sampling_cvar(
    arms: Dict[str, BayesianArm],
    lambda_risk: float = LAMBDA_RISK,
    alpha_cvar: float = ALPHA_CVAR,
) -> str:
    """
    Risk-aware TS-like policy based on CVaR utility.
    """
    utilities = {}
    for name, arm in arms.items():
        u = arm.utility_cvar(lambda_risk=lambda_risk, alpha_cvar=alpha_cvar, n_mc=5_000)
        utilities[name] = u
    return max(utilities.keys(), key=lambda k: utilities[k])
