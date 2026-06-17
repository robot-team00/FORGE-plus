"""Isaac Lab assembly environment implementation.

Wraps Isaac Lab's DirectRLEnv to implement BaseAssemblyEnv.
Requires isaaclab>=2.0.0 and a CUDA-capable GPU.

Import guard: this module only imports Isaac Lab at function call time
to allow the rest of the package to load without GPU/IsaacLab available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

from forge_plus.control.force_clamp import Wrench
from forge_plus.encoding.signature_encoder import ContactStep
from forge_plus.envs.base_assembly_env import (
    BaseAssemblyEnv,
    EnvObservation,
    EpisodeConfig,
    TaskOutcome,
    TaskPhase,
)


@dataclass
class IsaacLabEnvConfig:
    """Isaac Lab-specific configuration."""

    num_envs: int = 1
    headless: bool = True
    device: str = "cuda:0"
    physics_dt: float = 1.0 / 120.0      # 120 Hz physics
    control_decimation: int = 2           # control at 60 Hz
    contact_filter_hz: float = 30.0
    near_contact_vel_m_s: float = 0.005   # velocity limit when close to contact
    controller_stiffness: float = 100.0   # N/m — intentionally low for overshoot control
    controller_damping: float = 20.0


class IsaacLabAssemblyEnv(BaseAssemblyEnv):
    """Isaac Lab implementation of the assembly environment.

    Usage:
        env = IsaacLabAssemblyEnv(IsaacLabEnvConfig())
        obs = env.reset(episode_cfg)
        wrench = Wrench(0, 0, 10, 0, 0, 0)
        obs, outcome = env.step(wrench)
    """

    def __init__(self, cfg: IsaacLabEnvConfig) -> None:
        self._cfg = cfg
        self._sim = None
        self._robot = None
        self._object = None
        self._ft_sensor = None
        self._contact_sensor = None
        self._episode_cfg: EpisodeConfig | None = None
        self._step_count = 0
        self._phase = TaskPhase.APPROACH
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS
        self._history: list[ContactStep] = []
        self._last_ft: Wrench = Wrench(0, 0, 0, 0, 0, 0)
        self._insert_pos_mm = 0.0
        self._init_isaac()

    def _init_isaac(self) -> None:
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.envs import DirectRLEnvCfg
            from isaaclab.scene import InteractiveSceneCfg
        except ImportError as e:
            raise ImportError(
                "Isaac Lab is required for IsaacLabAssemblyEnv. "
                "Install with: pip install isaaclab>=2.0.0\n"
                f"Original error: {e}"
            ) from e

        # Isaac Lab simulation configuration
        # Full scene setup is done in reset() per episode config
        self._isaac_available = True

    def reset(self, episode_cfg: EpisodeConfig) -> EnvObservation:
        self._episode_cfg = episode_cfg
        self._step_count = 0
        self._phase = TaskPhase.APPROACH
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS
        self._history.clear()
        self._insert_pos_mm = 0.0

        # In a full implementation this would:
        # 1. Load the object asset from the registry
        # 2. Set up the robot (Franka or Robotiq based on episode_cfg.gripper)
        # 3. Run GraspGen to seed initial grasp pose
        # 4. Apply disturbance perturbation from episode_cfg.disturbance_seed
        # 5. Initialize F/T sensor
        # The hidden F_break is stored in self._episode_cfg.f_break_n and
        # NEVER passed to the skill, encoder, or LLM.

        return self._observe()

    def step(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        if self._done:
            return self._observe(), self._outcome

        import torch

        self._step_count += 1

        # In a full implementation this would:
        # 1. Convert wrench to joint torques via Jacobian
        # 2. Apply impedance controller
        # 3. Step Isaac Lab physics
        # 4. Read F/T sensor
        # 5. Check contact force against hidden F_break (evaluator only)

        # Evaluator-only breakage check (never exposed in observation)
        if self._episode_cfg:
            contact_f = self.get_contact_force_magnitude()
            if contact_f > self._episode_cfg.f_break_n:
                self._done = True
                self._outcome = TaskOutcome.BROKEN

        self._last_ft = wrench_cmd
        return self._observe(), self._outcome

    def apply_recovery(self, action: str, params: dict[str, Any]) -> None:
        """Execute a recovery primitive in Isaac Lab."""
        # Each action drives the robot via the controller:
        # retract_and_reapproach: move back N mm, re-approach
        # wiggle_search: sinusoidal lateral search pattern
        # rotate_align: rotate EE by params["yaw_deg"]
        # regrasp: open gripper, re-close on object
        # abort: no-op (episode runner handles the abort outcome)
        self._phase = TaskPhase.APPROACH
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS

    def observe(self) -> EnvObservation:
        """Return the current observation without resetting the scene."""
        return self._observe()

    def get_contact_force_magnitude(self) -> float:
        """Read peak contact force from the F/T sensor."""
        return float(np.linalg.norm([self._last_ft.fx, self._last_ft.fy, self._last_ft.fz]))

    def is_done(self) -> bool:
        return self._done

    @property
    def current_phase(self) -> TaskPhase:
        return self._phase

    def _observe(self) -> EnvObservation:
        cs = ContactStep(
            axial_force_n=abs(self._last_ft.fz),
            lateral_force_x_n=self._last_ft.fx,
            lateral_force_y_n=self._last_ft.fy,
            torque_z_nm=self._last_ft.tz,
            insert_pos_mm=self._insert_pos_mm,
            dt_ms=1000.0 * self._cfg.physics_dt * self._cfg.control_decimation,
        )
        return EnvObservation(
            joint_pos=np.zeros(7),
            joint_vel=np.zeros(7),
            ee_pos=np.zeros(3),
            ee_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            ft_wrench=self._last_ft,
            contact_step=cs,
            phase=self._phase,
            step_count=self._step_count,
        )
