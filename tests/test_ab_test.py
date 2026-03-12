from src.evaluation.ab_test_simulator import ABTestSimulator, ABTestResult
from src.agent.heuristic_baseline import HeuristicBaseline


def test_ab_simulator_runs():
    sim = ABTestSimulator(
        treatment_agent=HeuristicBaseline(),
        control_agent=HeuristicBaseline(),
        env_config={},
        n_simulations=5,
    )
    result = sim.run()
    assert hasattr(result, "mean_delta_margin")
    assert hasattr(result, "p_value")
    assert hasattr(result, "ci_95")


def test_ab_result_structure():
    sim = ABTestSimulator(
        treatment_agent=HeuristicBaseline(),
        control_agent=HeuristicBaseline(),
        env_config={},
        n_simulations=3,
    )
    result = sim.run()
    assert isinstance(result, ABTestResult)
    assert isinstance(result.mean_delta_margin, float)
    assert isinstance(result.p_value, float)
    assert len(result.ci_95) == 2
    assert len(result.treatment_margins) == 3
    assert len(result.control_margins) == 3


def test_ab_cumulative_curves():
    sim = ABTestSimulator(
        treatment_agent=HeuristicBaseline(),
        control_agent=HeuristicBaseline(),
        env_config={},
        n_simulations=5,
    )
    result = sim.run()
    assert "treatment" in result.cumulative_curves
    assert "control" in result.cumulative_curves
    assert len(result.cumulative_curves["treatment"]) == 5
