import numpy as np
from src.monitoring.drift_detector import DriftDetector


def test_drift_detector_creation():
    dd = DriftDetector()
    assert dd is not None
    alerts = dd.check_alerts()
    assert alerts["reward_drift"] is False
    assert alerts["action_entropy_collapse"] is False


def test_drift_detector_no_alerts_normal():
    """Normal operation should not trigger alerts."""
    dd = DriftDetector(config={"alert_consecutive_periods": 3})
    rng = np.random.default_rng(42)

    # Warmup + normal periods
    for period in range(20):
        for step in range(10):
            dd.update(
                reward=rng.normal(5.0, 1.0),
                action=int(rng.integers(0, 7)),
            )
        dd.end_period()

    alerts = dd.check_alerts()
    assert alerts["reward_drift"] is False


def test_drift_detector_reward_drift_alert():
    """Drastic reward change should trigger drift alert."""
    dd = DriftDetector(config={
        "alert_consecutive_periods": 3,
        "reward_drift_sigma": 2.0,
    })

    # Warmup with stable rewards
    for period in range(10):
        for step in range(10):
            dd.update(reward=5.0, action=int(np.random.randint(0, 7)))
        dd.end_period()

    # Now shift rewards dramatically
    for period in range(5):
        for step in range(10):
            dd.update(reward=50.0, action=int(np.random.randint(0, 7)))
        dd.end_period()

    alerts = dd.check_alerts()
    assert alerts["reward_drift"] is True


def test_drift_detector_entropy_collapse():
    """Using only one action should trigger entropy alert."""
    dd = DriftDetector(config={
        "alert_consecutive_periods": 3,
        "action_entropy_min": 0.5,
    })

    # Always take the same action
    for period in range(5):
        for step in range(20):
            dd.update(reward=1.0, action=0)  # always hold
        dd.end_period()

    alerts = dd.check_alerts()
    assert alerts["action_entropy_collapse"] is True


def test_drift_detector_report():
    dd = DriftDetector()
    for period in range(5):
        for step in range(10):
            dd.update(reward=float(period), action=period % 7)
        dd.end_period()

    report = dd.generate_report()
    assert "period_count" in report
    assert report["period_count"] == 5
    assert "alerts" in report
    assert "reward_history" in report
    assert len(report["reward_history"]) == 5


def test_drift_detector_elasticity_tracking():
    dd = DriftDetector()
    for period in range(5):
        for step in range(10):
            dd.update(
                reward=1.0, action=0,
                elasticity_observed=-1.5,
                elasticity_expected=-1.5,
            )
        dd.end_period()

    report = dd.generate_report()
    assert "elasticity_mae" in report
    assert report["elasticity_mae"] < 0.01  # very accurate
