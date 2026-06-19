"""
FrankaInsertionEnv — DirectRLEnv for Franka peg insertion.

Observation (34-dim, matches task1_franka_panda.pt checkpoint):
  joint_pos(7) + joint_vel(7) + ee_pos(3) + ee_quat(4) + ft_wrench(6) + phase_onehot(7)

Action (7-dim): delta EE pose [delta_pos(3), delta_quat(4)]
"""
from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Optional

import torch
import numpy as np

# -- Isaac Lab imports (must be after SimulationApp is started) --
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, ContactSensor


@configclass
class InsertionEnvCfg(DirectRLEnvCfg):
    """Configuration for Franka peg insertion env."""

    # -- Simulation --
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)

    # -- Episode --
    episode_length_s: float = 10.0  # 200 steps at 20Hz control

    # -- Spaces --
    observation_space: int = 34   # must match checkpoint
    action_space: int = 7         # delta_pos(3) + delta_quat(4)
    state_space: int = 0

    # -- Scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)

    # -- Asset paths (resolved at runtime) --
    # Robot: use FRANKA_PANDA_CFG if available, else fallback USD
    robot_usd: str = ""  # filled by _resolve_assets()

    # Control gains for delta-EE → joint torque
    action_scale: float = 0.01   # scale applied to raw delta actions
    max_delta_pos: float = 0.05  # m per step cap
    max_delta_angle: float = 0.1  # rad per step cap


class FrankaInsertionEnv(DirectRLEnv):
    """Isaac Lab DirectRLEnv for Franka peg insertion.

    Obs  = [joint_pos(7), joint_vel(7), ee_pos(3), ee_quat(4), ft_wrench(6), phase_onehot(7)]
    Act  = [delta_ee_pos(3), delta_ee_quat(4)]  (clamped)
    """

    cfg: InsertionEnvCfg

    # Phase constants (must match forge_plus TaskPhase enum)
    NUM_PHASES = 7

    def __init__(self, cfg: InsertionEnvCfg, render_mode: Optional[str] = None, **kwargs):
        # Resolve Franka USD path before super().__init__
        self._resolve_robot_usd(cfg)
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Buffers
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._ft_wrench = torch.zeros(self.num_envs, 6, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # EE body index (set after scene creation)
        self._ee_idx: int = -1
        self._ee_idx = self._robot.find_bodies("panda_hand")[0][0]

    # ------------------------------------------------------------------
    # Asset resolution
    # ------------------------------------------------------------------
    def _resolve_robot_usd(self, cfg: InsertionEnvCfg) -> None:
        """Try isaaclab_assets, fall back to Isaac Sim bundled USD."""
        try:
            from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # type: ignore
            self._franka_cfg = FRANKA_PANDA_CFG
            cfg.robot_usd = "isaaclab_assets"
        except ImportError:
            # Fall back: define inline ArticulationCfg using Isaac Sim bundled USD
            import os
            candidates = [
                "/workspace/FORGE-plus/assets/franka/franka.usd",
                "/workspace/IsaacLab/source/isaaclab_assets/data/Robots/FrankaEmika/panda.usd",
                "/isaac-sim/standalone_examples/api/omni.isaac.franka/usd/franka.usd",
                "/root/.local/share/ov/pkg/isaac_sim-*/standalone_examples/api/omni.isaac.franka/usd/franka.usd",
            ]
            import glob
            usd_path = None
            for c in candidates:
                found = glob.glob(c)
                if found:
                    usd_path = found[0]
                    break
            if usd_path is None:
                usd_path = "{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
            cfg.robot_usd = usd_path
            self._franka_cfg = None

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        """Create Franka + table + peg + socket."""
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        # ---- Robot ----
        if self._franka_cfg is not None:
            robot_cfg = self._franka_cfg.replace(
                prim_path="/World/envs/env_.*/Robot"
            )
            # PATCH: override USD to local file (Nucleus unavailable)
            robot_cfg.spawn.usd_path = "/workspace/FORGE-plus/assets/franka/panda_instanceable.usd"
            robot_cfg.spawn.activate_contact_sensors = True
        else:
            robot_cfg = ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=self.cfg.robot_usd,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        max_depenetration_velocity=5.0,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True,
                        solver_position_iteration_count=8,
                        solver_velocity_iteration_count=0,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    joint_pos={
                        "panda_joint1": 0.0,
                        "panda_joint2": -0.785,
                        "panda_joint3": 0.0,
                        "panda_joint4": -2.356,
                        "panda_joint5": 0.0,
                        "panda_joint6": 1.571,
                        "panda_joint7": 0.785,
                        "panda_finger_joint.*": 0.04,
                    },
                    pos=(0.0, 0.0, 0.4),  # on top of table
                ),
                actuators={
                    "panda_shoulder": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["panda_joint[1-4]"],
                        effort_limit=87.0,
                        velocity_limit=2.175,
                        stiffness=80.0,
                        damping=4.0,
                    ),
                    "panda_forearm": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["panda_joint[5-7]"],
                        effort_limit=12.0,
                        velocity_limit=2.61,
                        stiffness=80.0,
                        damping=4.0,
                    ),
                    "panda_hand": sim_utils.ImplicitActuatorCfg(
                        joint_names_expr=["panda_finger_joint.*"],
                        effort_limit=200.0,
                        velocity_limit=0.2,
                        stiffness=2000.0,
                        damping=1.0,
                    ),
                },
            )

        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # ---- Table ----
        table_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=sim_utils.CuboidCfg(
                size=(0.9, 0.6, 0.4),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.35, 0.2)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.2)),
        )
        self._table = RigidObject(table_cfg)
        self.scene.rigid_objects["table"] = self._table

        # ---- Socket (target) ----
        socket_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Socket",
            spawn=sim_utils.CuboidCfg(
                size=(0.06, 0.06, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.425)),
        )
        self._socket = RigidObject(socket_cfg)
        self.scene.rigid_objects["socket"] = self._socket

        # ---- Peg (manipulated object) ----
        peg_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Peg",
            spawn=sim_utils.CylinderCfg(
                radius=0.012,
                height=0.10,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.55)),
        )
        self._peg = RigidObject(peg_cfg)
        self.scene.rigid_objects["peg"] = self._peg

        # ---- Ground plane + lighting ----
        sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)).func(
            "/World/DomeLight", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        )

        # Clone + filter
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()

    def _apply_action(self) -> None:
        """Convert delta EE actions → joint position targets via simple Jacobian/IK fallback."""
        # Clamp actions
        delta_pos = self._actions[:, :3] * self.cfg.action_scale
        delta_pos = delta_pos.clamp(-self.cfg.max_delta_pos, self.cfg.max_delta_pos)
        # For simplicity, apply as joint velocity targets proportional to delta
        # In a full impl, use DifferentialIKController here
        # Use current joint positions + small proportional nudge
        joint_pos = self._robot.data.joint_pos[:, :7].clone()
        # Jacobian-free: nudge first 3 joints proportionally (placeholder for demo)
        joint_targets = joint_pos.clone()
        joint_targets[:, 0] += delta_pos[:, 0] * 0.5
        joint_targets[:, 1] += delta_pos[:, 2] * 0.5
        joint_targets[:, 2] += delta_pos[:, 1] * 0.3
        self._robot.set_joint_position_target(joint_targets, joint_ids=list(range(7)))

    def _get_observations(self) -> dict:
        """Return 34-dim obs tensor matching the trained checkpoint."""
        # joint_pos (7)
        joint_pos = self._robot.data.joint_pos[:, :7]
        # joint_vel (7)
        joint_vel = self._robot.data.joint_vel[:, :7]
        # ee_pos (3) + ee_quat (4)
        ee_pos_w = self._robot.data.body_pos_w[:, self._ee_idx, :]   # (N, 3)
        ee_quat_w = self._robot.data.body_quat_w[:, self._ee_idx, :]  # (N, 4)
        # ft_wrench (6) — simulated as zeros (no F/T sensor)
        ft_wrench = self._ft_wrench  # (N, 6) zeros
        # phase_onehot (7)
        phase_oh = torch.zeros(self.num_envs, self.NUM_PHASES, device=self.device)
        phase_oh.scatter_(1, self._phase.unsqueeze(1), 1.0)

        obs = torch.cat([joint_pos, joint_vel, ee_pos_w, ee_quat_w, ft_wrench, phase_oh], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        # Reset robot to default pose
        default_joint_pos = self._robot.data.default_joint_pos[env_ids]
        default_joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])
        self._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        self._phase[env_ids] = 0
        self._ft_wrench[env_ids] = 0.0
        self._step_count[env_ids] = 0


# -- Gymnasium registration --
import gymnasium as gym

gym.register(
    id="FORGE-Insertion-v0",
    entry_point="forge_plus.isaac_insertion_env:FrankaInsertionEnv",
    kwargs={"cfg": InsertionEnvCfg()},
)
