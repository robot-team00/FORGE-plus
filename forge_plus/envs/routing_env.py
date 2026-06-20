"""Task-routing environment.

The eval harness builds one env and reuses it across episodes that may span
several tasks (e.g. --task all). Task 3 (place/stack) needs a different
simulator from the insertion tasks, so this wrapper dispatches each reset() to
the right backend by task name and forwards every other call to the active one.
"""

from __future__ import annotations

from typing import Any

from forge_plus.control.force_clamp import Wrench
from forge_plus.envs.base_assembly_env import (
    BaseAssemblyEnv,
    EnvObservation,
    EpisodeConfig,
    TaskOutcome,
    TaskPhase,
)
from forge_plus.envs.mock_assembly_env import MockAssemblyEnv, MockEnvConfig
from forge_plus.envs.place_stack_env import PlaceStackEnv, PlaceStackEnvConfig

_PLACE_TASKS = ("task3", "task3_fragile_place")


def is_place_task(task_name: str) -> bool:
    return any(task_name.startswith(t) for t in _PLACE_TASKS)


class RoutingAssemblyEnv(BaseAssemblyEnv):
    """Routes to MockAssemblyEnv (insertion) or PlaceStackEnv (place/stack)."""

    def __init__(
        self,
        mock_cfg: MockEnvConfig | None = None,
        place_cfg: PlaceStackEnvConfig | None = None,
    ) -> None:
        self._mock = MockAssemblyEnv(mock_cfg or MockEnvConfig())
        self._place = PlaceStackEnv(place_cfg or PlaceStackEnvConfig())
        self._active: BaseAssemblyEnv = self._mock

    def reset(self, cfg: EpisodeConfig) -> EnvObservation:
        self._active = self._place if is_place_task(cfg.task_name) else self._mock
        return self._active.reset(cfg)

    def step(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        return self._active.step(wrench_cmd)

    def apply_recovery(self, action: str, params: dict[str, Any]) -> None:
        self._active.apply_recovery(action, params)

    def observe(self) -> EnvObservation:
        return self._active.observe()

    def get_contact_force_magnitude(self) -> float:
        return self._active.get_contact_force_magnitude()

    def is_done(self) -> bool:
        return self._active.is_done()

    def current_failure_mode(self) -> str | None:
        return self._active.current_failure_mode()

    @property
    def current_phase(self) -> TaskPhase:
        return self._active.current_phase
