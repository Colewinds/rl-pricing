"""Tests for ModelRegistry and ModelVersion."""

import json

import pytest

from src.pipeline.model_registry import ModelRegistry, ModelVersion


@pytest.fixture
def registry(tmp_path):
    """Create a ModelRegistry backed by a temporary directory."""
    return ModelRegistry(registry_dir=str(tmp_path))


def _register_one(registry, algorithm="ppo", timesteps=10_000):
    """Helper to register a single model with sensible defaults."""
    return registry.register_model(
        model_path="/fake/model.zip",
        algorithm=algorithm,
        training_timesteps=timesteps,
        eval_metrics={"mean_reward": 42.0},
    )


# -- register_model -----------------------------------------------------------

class TestRegisterModel:
    def test_returns_model_version(self, registry):
        version = _register_one(registry)
        assert isinstance(version, ModelVersion)

    def test_version_id_populated(self, registry):
        version = _register_one(registry)
        assert version.version_id.startswith("v1_")

    def test_fields_match_inputs(self, registry):
        version = registry.register_model(
            model_path="/models/ppo.zip",
            algorithm="ppo",
            training_timesteps=50_000,
            eval_metrics={"sharpe": 1.2},
            ab_test_result={"uplift": 0.05},
        )
        assert version.model_path == "/models/ppo.zip"
        assert version.algorithm == "ppo"
        assert version.training_timesteps == 50_000
        assert version.eval_metrics == {"sharpe": 1.2}
        assert version.ab_test_result == {"uplift": 0.05}

    def test_eval_metrics_defaults_to_empty_dict(self, registry):
        version = registry.register_model(
            model_path="/m.zip",
            algorithm="sac",
            training_timesteps=1_000,
        )
        assert version.eval_metrics == {}

    def test_ab_test_result_defaults_to_none(self, registry):
        version = _register_one(registry)
        assert version.ab_test_result is None

    def test_sequential_version_ids(self, registry):
        v1 = _register_one(registry)
        v2 = _register_one(registry)
        assert v1.version_id.startswith("v1_")
        assert v2.version_id.startswith("v2_")


# -- auto-champion on first registration -------------------------------------

class TestAutoChampion:
    def test_first_model_is_champion(self, registry):
        version = _register_one(registry)
        assert version.is_champion is True

    def test_second_model_is_not_champion(self, registry):
        _register_one(registry)
        v2 = _register_one(registry)
        assert v2.is_champion is False

    def test_only_one_champion_after_multiple_registers(self, registry):
        for _ in range(5):
            _register_one(registry)
        champions = [v for v in registry.list_versions() if v.is_champion]
        assert len(champions) == 1


# -- get_champion -------------------------------------------------------------

class TestGetChampion:
    def test_returns_none_when_empty(self, registry):
        assert registry.get_champion() is None

    def test_returns_first_model(self, registry):
        v1 = _register_one(registry)
        assert registry.get_champion().version_id == v1.version_id

    def test_returns_champion_after_multiple_registers(self, registry):
        v1 = _register_one(registry)
        _register_one(registry)
        _register_one(registry)
        assert registry.get_champion().version_id == v1.version_id


# -- promote_to_champion ------------------------------------------------------

class TestPromoteToChampion:
    def test_returns_true_on_success(self, registry):
        _register_one(registry)
        v2 = _register_one(registry)
        assert registry.promote_to_champion(v2.version_id) is True

    def test_promoted_version_becomes_champion(self, registry):
        _register_one(registry)
        v2 = _register_one(registry)
        registry.promote_to_champion(v2.version_id)
        assert registry.get_champion().version_id == v2.version_id

    def test_old_champion_is_demoted(self, registry):
        v1 = _register_one(registry)
        v2 = _register_one(registry)
        registry.promote_to_champion(v2.version_id)
        refreshed_v1 = registry.get_version(v1.version_id)
        assert refreshed_v1.is_champion is False

    def test_only_one_champion_after_promotion(self, registry):
        for _ in range(4):
            _register_one(registry)
        v5 = _register_one(registry)
        registry.promote_to_champion(v5.version_id)
        champions = [v for v in registry.list_versions() if v.is_champion]
        assert len(champions) == 1
        assert champions[0].version_id == v5.version_id

    def test_returns_false_for_unknown_version_id(self, registry):
        _register_one(registry)
        assert registry.promote_to_champion("nonexistent_id") is False

    def test_champion_unchanged_after_failed_promotion(self, registry):
        v1 = _register_one(registry)
        registry.promote_to_champion("nonexistent_id")
        assert registry.get_champion().version_id == v1.version_id

    def test_promote_already_champion_is_noop(self, registry):
        v1 = _register_one(registry)
        assert registry.promote_to_champion(v1.version_id) is True
        assert registry.get_champion().version_id == v1.version_id
        champions = [v for v in registry.list_versions() if v.is_champion]
        assert len(champions) == 1


# -- list_versions ------------------------------------------------------------

class TestListVersions:
    def test_empty_registry(self, registry):
        assert registry.list_versions() == []

    def test_returns_all_registered(self, registry):
        _register_one(registry)
        _register_one(registry)
        _register_one(registry)
        assert len(registry.list_versions()) == 3

    def test_returns_copy(self, registry):
        _register_one(registry)
        versions = registry.list_versions()
        versions.clear()
        assert len(registry.list_versions()) == 1


# -- get_version --------------------------------------------------------------

class TestGetVersion:
    def test_returns_correct_version(self, registry):
        v1 = _register_one(registry, algorithm="ppo")
        v2 = _register_one(registry, algorithm="sac")
        fetched = registry.get_version(v2.version_id)
        assert fetched.version_id == v2.version_id
        assert fetched.algorithm == "sac"

    def test_returns_none_for_missing(self, registry):
        assert registry.get_version("does_not_exist") is None


# -- persistence --------------------------------------------------------------

class TestPersistence:
    def test_registry_json_created_on_register(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path))
        _register_one(reg)
        assert (tmp_path / "registry.json").exists()

    def test_registry_json_is_valid_json(self, tmp_path):
        reg = ModelRegistry(registry_dir=str(tmp_path))
        _register_one(reg)
        with open(tmp_path / "registry.json") as f:
            data = json.load(f)
        assert "versions" in data
        assert "last_updated" in data

    def test_reload_preserves_versions(self, tmp_path):
        reg1 = ModelRegistry(registry_dir=str(tmp_path))
        v = _register_one(reg1)

        reg2 = ModelRegistry(registry_dir=str(tmp_path))
        assert len(reg2.list_versions()) == 1
        assert reg2.list_versions()[0].version_id == v.version_id

    def test_reload_preserves_champion(self, tmp_path):
        reg1 = ModelRegistry(registry_dir=str(tmp_path))
        v1 = _register_one(reg1)

        reg2 = ModelRegistry(registry_dir=str(tmp_path))
        assert reg2.get_champion().version_id == v1.version_id

    def test_reload_preserves_promotion(self, tmp_path):
        reg1 = ModelRegistry(registry_dir=str(tmp_path))
        _register_one(reg1)
        v2 = _register_one(reg1)
        reg1.promote_to_champion(v2.version_id)

        reg2 = ModelRegistry(registry_dir=str(tmp_path))
        assert reg2.get_champion().version_id == v2.version_id

    def test_reload_preserves_eval_metrics(self, tmp_path):
        reg1 = ModelRegistry(registry_dir=str(tmp_path))
        reg1.register_model(
            model_path="/m.zip",
            algorithm="ppo",
            training_timesteps=1_000,
            eval_metrics={"reward": 99.9, "sharpe": 2.1},
        )

        reg2 = ModelRegistry(registry_dir=str(tmp_path))
        assert reg2.list_versions()[0].eval_metrics == {"reward": 99.9, "sharpe": 2.1}

    def test_creates_registry_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "path"
        reg = ModelRegistry(registry_dir=str(nested))
        _register_one(reg)
        assert nested.exists()
        assert (nested / "registry.json").exists()
