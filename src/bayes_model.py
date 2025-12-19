# src/bayes_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from .config import M0, KAPPA0, ALPHA0, BETA0


@dataclass
class NIGParams:
    m: float
    kappa: float
    alpha: float
    beta: float


class BayesianArm:
    """
    One asset with NIG posterior over daily returns.

    Likelihood: r_t | μ, σ² ~ N(μ, σ²)
    Prior:      (μ, σ²) ~ NIG(m0, κ0, alpha_0, beta_0)
    """

    def __init__(
        self,
        name: str,
        m0: float = M0,
        kappa0: float = KAPPA0,
        alpha0: float = ALPHA0,
        beta0: float = BETA0,
    ):
        self.name = name

        # prior
        self.m0 = m0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # sufficient stats
        self.n = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0

        # posterior params start as prior
        self.params = NIGParams(m0, kappa0, alpha0, beta0)

    def _recompute_posterior(self) -> None:
        n = self.n
        if n == 0:
            self.params = NIGParams(self.m0, self.kappa0, self.alpha0, self.beta0)
            return

        xbar = self.sum_x / n
        S = self.sum_x2 - n * xbar**2  # sum of squared deviations

        kappa_n = self.kappa0 + n
        m_n = (self.kappa0 * self.m0 + n * xbar) / kappa_n
        alpha_n = self.alpha0 + 0.5 * n
        beta_n = (
            self.beta0
            + 0.5 * S
            + (self.kappa0 * n * (xbar - self.m0) ** 2) / (2.0 * kappa_n)
        )

        self.params = NIGParams(m_n, kappa_n, alpha_n, beta_n)

    def update(self, x: float) -> None:
        """Observe one new return x and update posterior."""
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x * x
        self._recompute_posterior()

    def posterior_mean_mu(self) -> float:
        return self.params.m

    def posterior_mean_sigma2(self) -> float:
        # IG(α, β) mean is β / (α - 1) for α > 1
        a = self.params.alpha
        b = self.params.beta
        if a <= 1.0:
            return b / max(a - 1e-6, 1e-6)
        return b / (a - 1.0)

    def sample_posterior_mu_sigma2(self, size: int = 1):
        """
        Draw samples from joint posterior of (μ, σ²).
        σ² ~ IG(α, β) simulated via 1 / Gamma(α, 1/β)
        μ | σ² ~ N(m, σ² / κ)
        """
        a = self.params.alpha
        b = self.params.beta
        m = self.params.m
        kappa = self.params.kappa

        gamma_samples = np.random.gamma(shape=a, scale=1.0 / b, size=size)
        sigma2_samples = 1.0 / gamma_samples
        mu_samples = np.random.normal(
            loc=m, scale=np.sqrt(sigma2_samples / kappa), size=size
        )
        return mu_samples, sigma2_samples

    def sample_predictive(self, size: int = 1) -> np.ndarray:
        """Samples from posterior predictive r_new."""
        mu_s, sigma2_s = self.sample_posterior_mu_sigma2(size=size)
        return np.random.normal(loc=mu_s, scale=np.sqrt(sigma2_s), size=size)

    def posterior_risk_metrics(
        self,
        alpha_cvar: float = 0.05,
        n_mc: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Posterior predictive risk metrics:

        - mean_ret
        - std_ret
        - var_alpha: α-quantile of returns (VaR in return space)
        - cvar_loss: expected loss in worst α tail (positive number)
        """
        samples = self.sample_predictive(size=n_mc)
        mean_ret = float(samples.mean())
        std_ret = float(samples.std(ddof=1))

        q_alpha = float(np.quantile(samples, alpha_cvar))
        tail = samples[samples <= q_alpha]
        if tail.size == 0:
            cvar_loss = 0.0
        else:
            cvar_loss = float(-tail.mean())

        return {
            "mean_ret": mean_ret,
            "std_ret": std_ret,
            "var_alpha": q_alpha,
            "cvar_loss": cvar_loss,
        }

    def utility_mean_variance(self, lambda_risk: float, n_mc: int = 5_000) -> float:
        metrics = self.posterior_risk_metrics(n_mc=n_mc)
        return metrics["mean_ret"] - lambda_risk * metrics["std_ret"]

    def utility_cvar(
        self,
        lambda_risk: float,
        alpha_cvar: float = 0.05,
        n_mc: int = 5_000,
    ) -> float:
        metrics = self.posterior_risk_metrics(alpha_cvar=alpha_cvar, n_mc=n_mc)
        return metrics["mean_ret"] - lambda_risk * metrics["cvar_loss"]
