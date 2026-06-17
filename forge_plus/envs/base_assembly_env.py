"""Base assembly environment interface — abstract over Isaac Lab.

Concrete subclasses provide the Isaac Lab implementation.
This module defines the abstract API so the episode runner and skill code
can be tested independently of the GPU simulation stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from forge_plus.control.force_clamp import Wrench
from forge_plus.encoding.signature_encoder import ContactStep


class TaskPhase(Enum):
    APPROACH = "approach"
    INSERT = "insert"
    SEAT = "seat"
    THREAD = "thread"
    MESH = "mesh"
    CONTACT = "contact"
    SETTLE = "settle"


class TaskOutcome(Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE_STUCK = "failure_stuck"
    FAILURE_TIMEOUT = "failure_timeout"
    BROKEN = "broken"           # hidden F_break exceeded — evaluator-only label


@dataclass
class EnvObservation:
    """Sensor snapshot at one control step."""

    joint_pos: np.ndarray        # robot joint positions (rad)
    joint_vel: np.ndarray        # robot joint velocities (rad/s)
    ee_pos: np.ndarray           # end-effector position (m)
    ee_quat: np.ndarray          # end-effector quaternion [w, x, y, z]
    ft_wrench: Wrench            # force/torque at EE (N, N·m)
    contact_step: ContactStep    # pre-packaged for signature encoder
    phase: TaskPhase
    step_count: int


@dataclass
class EpisodeConfig:
    """Per-episode configuration."""

    object_key: str
    task_name: str
    gripper: str                 # "franka_panda" | "robotiq_2f140"
    f_break_n: float             # HIDDEN — set by evaluator, never exposed to agent
    max_steps: int = 2000
    disturbance_seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class BaseAssemblyEnv(ABC):
    """Abstract assembly environment.

    Concrete Isaac Lab environments inherit from this and implement the
    abstract methods. This layer decouples the episode runner from the sim.
    """

    @abstractmethod
    def reset(self, cfg: EpisodeConfig) -> EnvObservation:
        """Reset the environment and return the initial observation."""
        ...

    @abstractmethod
    def step(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        """Apply a wrench command, advance simulation, return (obs, outcome).

        The implementation checks peak_contact_force against f_break internally
        for the evaluator's breakage log, but does NOT expose f_break to the
        skill or the episode runner via the observation.
        """
        ...

    @abstractmethod
    def apply_recovery(self, action: str, params: dict[str, Any]) -> None:
        """Execute a recovery action (retract, wiggle, etc.) within the sim."""
        ...

    @abstractmethod
    def get_contact_force_magnitude(self) -> float:
        """Return the last measured peak contact force (N) — observable."""
        ...

    @abstractmethod
    def is_done(self) -> bool:
        """True if the task sub-phase is complete (success or stuck)."""
        ...

    @property
    @abstractmethod
    def current_phase(self) -> TaskPhase:
        ...
