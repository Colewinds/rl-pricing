"""Model registry for champion/challenger model management."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ModelVersion:
    """Metadata for a trained model version."""
    version_id: str
    model_path: str
    algorithm: str
    trained_at: str
    training_timesteps: int
    eval_metrics: dict = field(default_factory=dict)
    ab_test_result: dict | None = None
    is_champion: bool = False


class ModelRegistry:
    """Tracks champion/challenger models with metadata.

    Stores a manifest at results/models/registry.json.
    """

    def __init__(self, registry_dir: str = "results/models"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.registry_dir / "registry.json"
        self._versions: list[ModelVersion] = []
        self._load()

    def _load(self):
        """Load existing registry from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                data = json.load(f)
            self._versions = [ModelVersion(**v) for v in data.get("versions", [])]

    def _save(self):
        """Persist registry to disk."""
        data = {
            "versions": [asdict(v) for v in self._versions],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register_model(
        self,
        model_path: str,
        algorithm: str,
        training_timesteps: int,
        eval_metrics: dict | None = None,
        ab_test_result: dict | None = None,
    ) -> ModelVersion:
        """Register a newly trained model.

        Args:
            model_path: Path to the saved model.
            algorithm: Algorithm used (e.g., "ppo").
            training_timesteps: Number of timesteps trained.
            eval_metrics: Evaluation metrics dict.
            ab_test_result: A/B test results if available.

        Returns:
            The registered ModelVersion.
        """
        version_id = f"v{len(self._versions) + 1}_{datetime.now():%Y%m%d_%H%M%S}"
        is_champion = len(self._versions) == 0  # first model is auto-champion

        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            algorithm=algorithm,
            trained_at=datetime.now().isoformat(),
            training_timesteps=training_timesteps,
            eval_metrics=eval_metrics or {},
            ab_test_result=ab_test_result,
            is_champion=is_champion,
        )
        self._versions.append(version)
        self._save()
        return version

    def get_champion(self) -> ModelVersion | None:
        """Get the current champion model."""
        for v in reversed(self._versions):
            if v.is_champion:
                return v
        return None

    def promote_to_champion(self, version_id: str) -> bool:
        """Promote a model version to champion, demoting the current one.

        Args:
            version_id: ID of the version to promote.

        Returns:
            True if promotion succeeded.
        """
        target = None
        for v in self._versions:
            if v.version_id == version_id:
                target = v

        if target is None:
            return False

        # Demote current champion
        for v in self._versions:
            v.is_champion = False

        target.is_champion = True
        self._save()
        return True

    def list_versions(self) -> list[ModelVersion]:
        """List all registered model versions."""
        return list(self._versions)

    def get_version(self, version_id: str) -> ModelVersion | None:
        """Get a specific version by ID."""
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None
