"""Regression guard: the names imported by examples/*.py must remain importable.

If a refactor renames or moves any of these, the corresponding example breaks.
This is intentionally just an import-smoke test — semantics are covered by the
other test modules.
"""

from __future__ import annotations


def test_top_level_imports():
    from smc_renewal.config import default_config  # noqa: F401
    from smc_renewal.state import (  # noqa: F401
        ParticleParams,
        ParticleState,
        state_dim,
        unpack,
    )
    from smc_renewal.synthetic import sample_initial_state, simulate  # noqa: F401
    from smc_renewal.transition import (  # noqa: F401
        TransitionNoise,
        expected_observation,
        step,
    )
    from smc_renewal.scoring import (  # noqa: F401
        crps_weighted_batch,
        interval_coverage,
        weighted_quantile,
    )
    from smc_renewal.forecast import forecast_from_cloud  # noqa: F401
    from smc_renewal.rolling_origin import (  # noqa: F401
        horizon_summaries,
        rolling_origin_forecast,
    )


def test_pf_runner_imports():
    from smc_renewal.pf.runner import run_liu_west  # noqa: F401
    from smc_renewal.pf.runner_sigma import run_liu_west_sigma  # noqa: F401
    from smc_renewal.pf.runner_trend import run_liu_west_trend  # noqa: F401
    from smc_renewal.pf.runner_discrete import run_liu_west_discrete  # noqa: F401
    from smc_renewal.pf.update import extend_liu_west  # noqa: F401


def test_pf_model_variant_imports():
    from smc_renewal.pf.model_sigma import (  # noqa: F401
        ParticleParamsSigma,
        ParticleStateSigma,
        TransitionNoiseSigma,
        expected_observation_sigma,
        step_sigma,
    )
    from smc_renewal.pf.model_trend import (  # noqa: F401
        ParticleParamsTrend,
        ParticleStateTrend,
        TransitionNoiseTrend,
        expected_observation_trend,
        step_trend,
    )
    from smc_renewal.pf.model_discrete import (  # noqa: F401
        ParticleParamsDiscrete,
        ParticleStateDiscrete,
        expected_observation_discrete,
        step_discrete,
    )


def test_smc2_imports():
    from smc_renewal.smc2.runner import extend_fit, fit_smc2  # noqa: F401
    from smc_renewal.smc2.ekf_cuthbert import (  # noqa: F401
        filter_trajectory,
        marginal_log_likelihood,
    )
