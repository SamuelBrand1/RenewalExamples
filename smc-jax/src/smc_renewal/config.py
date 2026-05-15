"""Model configuration for the nested-RW renewal SMC project."""

from __future__ import annotations

from dataclasses import dataclass

from jax import Array


@dataclass(frozen=True)
class ModelConfig:
    """Static (non-inferred) model configuration.

    Fields here are fixed at model-build time and shared across all particles.

    The feedback convolution uses the generation interval (no separate
    feedback PMF). The feedback magnitude is ``F = exp(log_F) > 0`` so the
    feedback always damps: ``Rt_eff = Rt_raw * exp(-F * conv(I, g))``.
    Priors on ``log_F`` follow the same nested-RW structure as ``log_Rt``.
    """

    generation_interval: Array
    delay_pmf: Array
    sigma_floor: float = 1e-4
    liu_west_h: float = 0.1
    init_log_Rt_mean: float = 0.0
    init_log_Rt_sd: float = 0.3
    init_log_sigma_R_mean: float = -3.5
    init_log_sigma_R_sd: float = 0.3
    # log_F prior — by default tuned for very weak feedback (F ≈ 6e-6).
    # Wider broadening is appropriate when γ-style feedback is expected to
    # vary across seasons (see example 07).
    init_log_F_mean: float = -12.0
    init_log_F_sd: float = 1.0
    init_log_sigma_F_mean: float = -12.0
    init_log_sigma_F_sd: float = 0.5
    init_log_I0_mean: float = 2.3
    init_log_I0_sd: float = 1.0
    prior_log_tau_R_mean: float = -4.0
    prior_log_tau_R_sd: float = 0.3
    prior_log_tau_F_mean: float = -12.0
    prior_log_tau_F_sd: float = 0.5
    prior_log_phi_mean: float = 2.3
    prior_log_phi_sd: float = 0.5
    # Model C (integrated-Brownian-motion) priors on the velocity state and
    # its volatility hyperparameters.  Defaults are conservative: small
    # starting velocity (no strong "trend" prior) and modest velocity-volatility
    # so the trend itself drifts slowly.
    init_v_R_mean: float = 0.0
    init_v_R_sd: float = 0.01
    init_v_F_mean: float = 0.0
    init_v_F_sd: float = 1e-4
    init_log_sigma_vR_mean: float = -5.0
    init_log_sigma_vR_sd: float = 1.0
    init_log_sigma_vF_mean: float = -9.0
    init_log_sigma_vF_sd: float = 1.0
    # Model D (discrete Poisson renewal) — immigration rate μ inferred via
    # Liu-West.  Default mean log μ = 0 ⇒ μ ≈ 1.0 case/day; wide-ish sd.
    init_log_mu_mean: float = 0.0
    init_log_mu_sd: float = 0.5

    @property
    def Tg(self) -> int:
        return int(self.generation_interval.shape[0])

    @property
    def Td(self) -> int:
        return int(self.delay_pmf.shape[0])

    @property
    def buffer_len(self) -> int:
        return max(self.Tg, self.Td)


def default_config() -> ModelConfig:
    """Default config with covid/flu-like discretized PMFs."""
    from smc_renewal.pmfs import (
        default_delay_pmf,
        default_generation_interval,
    )

    return ModelConfig(
        generation_interval=default_generation_interval(),
        delay_pmf=default_delay_pmf(),
    )
