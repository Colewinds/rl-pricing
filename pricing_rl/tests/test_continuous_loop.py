"""Tests for the ContinuousLoop continuous learning pipeline.

Mocks _train_agent and _evaluate_model to test loop logic,
drift-triggered retrain, scheduled retrain, and promotion decisions
without actual RL training.
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from src.evaluation.ab_test_simulator import ABTestResult
from src.pipeline.continuous_loop import ContinuousLoop
from src.pipeline.model_registry import ModelVersion


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_champion(version_id: str = "v1_champ", path: str = "/tmp/champ") -> ModelVersion:
    return ModelVersion(
        version_id=version_id,
        model_path=path,
        algorithm="ppo",
        trained_at="2025-01-01T00:00:00",
        training_timesteps=200000,
        eval_metrics={"mean_reward": 5.0},
        is_champion=True,
    )


def _make_ab_result(p_value: float, mean_delta: float) -> ABTestResult:
    return ABTestResult(
        mean_delta_margin=mean_delta,
        ci_95=(mean_delta - 1.0, mean_delta + 1.0),
        p_value=p_value,
        power=0.8,
        treatment_margins=[10.0] * 10,
        control_margins=[9.0] * 10,
    )


def _fake_eval_metrics(mean: float = 5.0, n: int = 50) -> dict:
    return {
        "mean_reward": mean,
        "std_reward": 1.0,
        "episode_rewards": [mean] * n,
    }


# ---------------------------------------------------------------------------
# 1. Initialization from config
# ---------------------------------------------------------------------------

@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_init_from_config(mock_clv, mock_drift, mock_registry):
    """ContinuousLoop should pick up values from the config dict."""
    config = _load_config()
    loop = ContinuousLoop(config=config, algorithm="ppo")

    assert loop.algorithm == "ppo"
    assert loop.retrain_interval == 4
    assert loop.retrain_timesteps == 200000
    assert loop.eval_episodes == 50
    assert loop.ab_simulations == 100
    assert loop.p_threshold == 0.05
    assert loop.warm_start is True
    assert loop._period == 0
    assert loop._history == []


@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_init_defaults_when_config_empty(mock_clv, mock_drift, mock_registry):
    """Missing keys should fall back to hardcoded defaults."""
    loop = ContinuousLoop(config={}, algorithm="dqn")

    assert loop.algorithm == "dqn"
    assert loop.retrain_interval == 4
    assert loop.retrain_timesteps == 200000
    assert loop.eval_episodes == 50


# ---------------------------------------------------------------------------
# 2. Loop logic with mocked training / evaluation
# ---------------------------------------------------------------------------

@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_run_period_no_retrain(mock_clv, mock_drift_cls, mock_registry_cls):
    """Period 1 with no drift should not trigger retrain."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion

    # No drift
    loop.drift_detector.check_alerts.return_value = {"any_alert": False}

    # Mock _evaluate_model to skip real env interaction
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())

    loop._period = 1  # period 1 -- not on retrain interval and no drift
    info = loop._run_period()

    assert info["retrained"] is False
    assert info["promoted"] is False
    loop._evaluate_model.assert_called_once_with(champion)
    loop.drift_detector.end_period.assert_called_once()
    loop.drift_detector.check_alerts.assert_called_once()


@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_run_calls_callback(mock_clv, mock_drift_cls, mock_registry_cls):
    """run() should invoke the callback each period."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    loop.registry.get_champion.return_value = None
    loop.drift_detector.check_alerts.return_value = {"any_alert": False}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())

    callback = MagicMock()
    history = loop.run(max_periods=3, callback=callback)

    assert callback.call_count == 3
    assert len(history) == 3


# ---------------------------------------------------------------------------
# 3. Drift-triggered retrain
# ---------------------------------------------------------------------------

@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_drift_triggered_retrain(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """any_alert=True from drift detector should trigger retraining."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion

    # Drift detected
    loop.drift_detector.check_alerts.return_value = {"any_alert": True}

    # Mock evaluation and training
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics(mean=6.0))

    challenger_version = ModelVersion(
        version_id="v2_challenger",
        model_path="/tmp/challenger",
        algorithm="ppo",
        trained_at="2025-01-02T00:00:00",
        training_timesteps=200000,
    )
    loop.registry.register_model.return_value = challenger_version

    fake_agent = MagicMock()
    mock_train.return_value = (fake_agent, "/tmp/challenger")

    # Mock A/B test -- challenger wins
    loop._ab_test = MagicMock(return_value=_make_ab_result(p_value=0.01, mean_delta=2.0))

    loop._period = 1  # not on scheduled interval, but drift forces retrain
    info = loop._run_period()

    assert info["retrained"] is True
    mock_train.assert_called_once()
    loop._ab_test.assert_called_once_with(champion.model_path, "/tmp/challenger")
    assert info["promoted"] is True


@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_drift_retrain_uses_warm_start(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """Drift-triggered retrain should warm-start from champion when enabled."""
    config = _load_config()
    loop = ContinuousLoop(config=config)
    assert loop.warm_start is True

    champion = _make_champion(path="/tmp/champ_model")
    loop.registry.get_champion.return_value = champion
    loop.drift_detector.check_alerts.return_value = {"any_alert": True}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics())

    challenger_version = _make_champion(version_id="v2", path="/tmp/new")
    loop.registry.register_model.return_value = challenger_version
    loop._ab_test = MagicMock(return_value=_make_ab_result(p_value=0.01, mean_delta=1.0))

    mock_train.return_value = (MagicMock(), "/tmp/new")

    loop._period = 1
    loop._run_period()

    # Verify warm_start_path was passed
    _, kwargs = mock_train.call_args
    assert kwargs["warm_start_path"] == "/tmp/champ_model"


# ---------------------------------------------------------------------------
# 4. Scheduled retrain at interval
# ---------------------------------------------------------------------------

@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_scheduled_retrain_at_interval(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """Retrain should trigger at retrain_interval periods even without drift."""
    config = _load_config()
    loop = ContinuousLoop(config=config)
    assert loop.retrain_interval == 4

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion
    loop.drift_detector.check_alerts.return_value = {"any_alert": False}  # no drift
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics())

    challenger_version = _make_champion(version_id="v2", path="/tmp/c2")
    loop.registry.register_model.return_value = challenger_version
    loop._ab_test = MagicMock(return_value=_make_ab_result(p_value=0.01, mean_delta=1.0))
    mock_train.return_value = (MagicMock(), "/tmp/c2")

    # Period 4 should trigger (period > 0 and period % 4 == 0)
    loop._period = 4
    info = loop._run_period()
    assert info["retrained"] is True
    mock_train.assert_called_once()


@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_no_retrain_between_intervals(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """Periods between intervals with no drift should NOT retrain."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion
    loop.drift_detector.check_alerts.return_value = {"any_alert": False}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())

    for period in [1, 2, 3, 5, 6, 7]:
        loop._period = period
        info = loop._run_period()
        assert info["retrained"] is False, f"Should not retrain at period {period}"

    mock_train.assert_not_called()


@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_scheduled_retrain_at_multiple_intervals(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """Retrain should trigger at every multiple of retrain_interval."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion
    loop.drift_detector.check_alerts.return_value = {"any_alert": False}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics())

    challenger_version = _make_champion(version_id="v2", path="/tmp/c2")
    loop.registry.register_model.return_value = challenger_version
    loop._ab_test = MagicMock(return_value=_make_ab_result(p_value=0.01, mean_delta=1.0))
    mock_train.return_value = (MagicMock(), "/tmp/c2")

    for period in [4, 8, 12]:
        loop._period = period
        info = loop._run_period()
        assert info["retrained"] is True, f"Should retrain at period {period}"

    assert mock_train.call_count == 3


# ---------------------------------------------------------------------------
# 5. Promotion when challenger wins (p < 0.05, positive delta)
# ---------------------------------------------------------------------------

@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_promotion_on_significant_improvement(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """Challenger with p < threshold and positive delta should be promoted."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion
    loop.drift_detector.check_alerts.return_value = {"any_alert": True}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics(mean=7.0))

    challenger_version = _make_champion(version_id="v2_win", path="/tmp/winner")
    loop.registry.register_model.return_value = challenger_version

    # Significant improvement
    ab_result = _make_ab_result(p_value=0.02, mean_delta=3.5)
    loop._ab_test = MagicMock(return_value=ab_result)

    mock_train.return_value = (MagicMock(), "/tmp/winner")

    loop._period = 1
    info = loop._run_period()

    assert info["promoted"] is True
    loop.registry.promote_to_champion.assert_called_once_with("v2_win")
    assert info["ab_result"]["p_value"] == 0.02
    assert info["ab_result"]["mean_delta"] == 3.5


@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_promotion_auto_when_no_champion(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """First model (no existing champion) should be auto-promoted."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    loop.registry.get_champion.return_value = None  # no champion yet
    loop.drift_detector.check_alerts.return_value = {"any_alert": True}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics())

    first_version = _make_champion(version_id="v1_first", path="/tmp/first")
    loop.registry.register_model.return_value = first_version

    mock_train.return_value = (MagicMock(), "/tmp/first")

    loop._period = 1
    info = loop._run_period()

    assert info["promoted"] is True
    loop.registry.promote_to_champion.assert_called_once_with("v1_first")


# ---------------------------------------------------------------------------
# 6. No promotion when p > threshold
# ---------------------------------------------------------------------------

@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_no_promotion_high_p_value(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """Challenger with p > threshold should NOT be promoted."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion
    loop.drift_detector.check_alerts.return_value = {"any_alert": True}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics())

    challenger_version = _make_champion(version_id="v2_lose", path="/tmp/loser")
    loop.registry.register_model.return_value = challenger_version

    # NOT significant: p = 0.45
    ab_result = _make_ab_result(p_value=0.45, mean_delta=0.5)
    loop._ab_test = MagicMock(return_value=ab_result)

    mock_train.return_value = (MagicMock(), "/tmp/loser")

    loop._period = 1
    info = loop._run_period()

    assert info["retrained"] is True
    assert info["promoted"] is False
    loop.registry.promote_to_champion.assert_not_called()


@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_no_promotion_negative_delta(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """Even low p-value should not promote if delta is negative (challenger worse)."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion
    loop.drift_detector.check_alerts.return_value = {"any_alert": True}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics())

    challenger_version = _make_champion(version_id="v2_worse", path="/tmp/worse")
    loop.registry.register_model.return_value = challenger_version

    # Low p but negative delta -- challenger is significantly *worse*
    ab_result = _make_ab_result(p_value=0.01, mean_delta=-2.0)
    loop._ab_test = MagicMock(return_value=ab_result)

    mock_train.return_value = (MagicMock(), "/tmp/worse")

    loop._period = 1
    info = loop._run_period()

    assert info["retrained"] is True
    assert info["promoted"] is False
    loop.registry.promote_to_champion.assert_not_called()


@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_no_promotion_borderline_p_value(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """p == threshold (exactly 0.05) should not promote (condition is strict <)."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion
    loop.drift_detector.check_alerts.return_value = {"any_alert": True}
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics())

    challenger_version = _make_champion(version_id="v2_border", path="/tmp/border")
    loop.registry.register_model.return_value = challenger_version

    # Exactly at threshold
    ab_result = _make_ab_result(p_value=0.05, mean_delta=1.0)
    loop._ab_test = MagicMock(return_value=ab_result)

    mock_train.return_value = (MagicMock(), "/tmp/border")

    loop._period = 4
    info = loop._run_period()

    assert info["retrained"] is True
    assert info["promoted"] is False
    loop.registry.promote_to_champion.assert_not_called()


# ---------------------------------------------------------------------------
# Integration-style: multiple periods via run()
# ---------------------------------------------------------------------------

@patch("src.pipeline.continuous_loop._train_agent")
@patch("src.pipeline.continuous_loop.ModelRegistry")
@patch("src.pipeline.continuous_loop.DriftDetector")
@patch("src.pipeline.continuous_loop.CLVOptimizer")
def test_run_multiple_periods_mixed(mock_clv, mock_drift_cls, mock_registry_cls, mock_train):
    """Run 8 periods: retrain should fire at periods 4 and on drift."""
    config = _load_config()
    loop = ContinuousLoop(config=config)

    champion = _make_champion()
    loop.registry.get_champion.return_value = champion
    loop._evaluate_model = MagicMock(return_value=_fake_eval_metrics())
    loop._evaluate_model_from_path = MagicMock(return_value=_fake_eval_metrics())

    challenger_version = _make_champion(version_id="v2", path="/tmp/v2")
    loop.registry.register_model.return_value = challenger_version
    loop._ab_test = MagicMock(return_value=_make_ab_result(p_value=0.10, mean_delta=0.5))
    mock_train.return_value = (MagicMock(), "/tmp/v2")

    # Drift on period 2 only; no drift otherwise
    call_count = {"n": 0}

    def alerts_side_effect():
        period = call_count["n"]
        call_count["n"] += 1
        if period == 2:
            return {"any_alert": True}
        return {"any_alert": False}

    loop.drift_detector.check_alerts.side_effect = alerts_side_effect

    history = loop.run(max_periods=8)

    assert len(history) == 8

    retrained_periods = [h["period"] for h in history if h["retrained"]]
    # Period 2 (drift) and period 4 (scheduled)
    assert 2 in retrained_periods
    assert 4 in retrained_periods
    # Period 0 does not retrain (period % 4 == 0 but period == 0 is excluded)
    assert 0 not in retrained_periods
