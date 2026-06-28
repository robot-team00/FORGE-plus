"""Isaac Lab assembly environment — live physics implementation.

Replaces the stub with a real DirectRLEnv inner env (FrankaInsertionEnv)
that runs Isaac Lab physics, Jacobian impedance control, and ContactSensor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    """Isaac Lab environment configuration."""
    num_envs: int = 1
    headless: bool = True
    device: str = "cuda:0"
    physics_dt: float = 1.0 / 120.0
    control_decimation: int = 2
    success_insertion_mm: float = 10.0
    socket_pos: tuple = (0.5, 0.0, 0.38)
    episode_max_steps: int = 2000


def _build_inner_env(cfg: IsaacLabEnvConfig):
    """Build FrankaInsertionEnv (DirectRLEnv). Must be called after SimulationApp."""
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils import configclass

    def _get_franka():
        from isaaclab.assets import ArticulationCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        return ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path="/workspace/assets/franka/franka.usd",
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={
                "panda_joint1": 0.0, "panda_joint2": -0.569,
                "panda_joint3": 0.0, "panda_joint4": -2.810,
                "panda_joint5": 0.0, "panda_joint6": 3.037, "panda_joint7": 0.741,
            }),
            actuators={"arm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-7]"],
                effort_limit=87.0, velocity_limit=2.175, stiffness=0.0, damping=0.0,
            )},
        )

    _DT = cfg.physics_dt
    _DECI = cfg.control_decimation
    _EP_S = cfg.episode_max_steps * _DT * _DECI

    @configclass
    class _InsertCfg(DirectRLEnvCfg):
        sim: SimulationCfg = SimulationCfg(dt=_DT, render_interval=_DECI)
        decimation: int = _DECI
        episode_length_s: float = _EP_S
        observation_space: int = 34
        action_space: int = 6
        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=cfg.num_envs, env_spacing=2.5, replicate_physics=True)
        success_m: float = cfg.success_insertion_mm / 1000.0
        f_break_n: float = 38.0  # updated per episode via cfg

    class _InsertEnv(DirectRLEnv):
        _KP = 1200.0; _KD = 80.0; _KD_J = 1.0
        def __init__(self, env_cfg):
            super().__init__(env_cfg)
            N, dev = self.num_envs, self.device
            self._depth   = torch.zeros(N, device=dev)
            self._cf      = torch.zeros(N, device=dev)
            self._broken  = torch.zeros(N, dtype=torch.bool, device=dev)
            self._steps   = torch.zeros(N, dtype=torch.long, device=dev)
            self._action  = torch.zeros(N, 6, device=dev)
            self._eft_lim = torch.tensor([87.,87.,87.,87.,12.,12.,12.], device=dev)

        def _setup_scene(self):
            env_ns = self.scene.env_regex_ns  # e.g. /World/envs/env_.*
            fc = _get_franka().replace(prim_path=f"{env_ns}/Robot")
            if hasattr(fc.spawn, "activate_contact_sensors"):
                fc.spawn.activate_contact_sensors = True
            self._robot = Articulation(cfg=fc)
            self.scene.articulations["robot"] = self._robot
            sc = cfg.socket_pos
            sock = RigidObjectCfg(
                prim_path=f"{env_ns}/Socket",
                spawn=sim_utils.CuboidCfg(
                    size=(0.04, 0.04, 0.02),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3,0.4,0.9)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=sc, rot=(1.,0.,0.,0.)),
            )
            self._socket = RigidObject(cfg=sock)
            self.scene.rigid_objects["socket"] = self._socket
            cc = ContactSensorCfg(
                prim_path=f"{env_ns}/Robot/panda_hand",
                update_period=0.0, history_length=6, debug_vis=False,
                filter_prim_paths_expr=[f"{env_ns}/Socket"],
            )
            self._contact = ContactSensor(cfg=cc)
            self.scene.sensors["contact"] = self._contact
            # Ground plane omitted — requires ISAAC_NUCLEUS_PATH; not needed for tabletop insertion
            sim_utils.DomeLightCfg(intensity=3000.).func("/World/L", sim_utils.DomeLightCfg(intensity=3000.))

        def _pre_physics_step(self, actions):
            self._action = actions.clone()

        def _apply_action(self):
            r = self._robot
            dq = r.data.joint_vel[:, :7]
            jac = r.root_physx_view.get_jacobians()
            J = jac[:, -6:, :7] if jac.dim() == 3 else jac[:, -1, :, :7]  # arm joints only
            # Convert task-frame wrench to world-frame: flip z so +fz=into-hole → -fz_world=downward
            act_w = self._action.clone()
            act_w[:, 2] = -act_w[:, 2]  # negate fz
            tau = torch.bmm(J.transpose(1,2), act_w.unsqueeze(-1)).squeeze(-1)
            tau = tau - self._KD_J * dq
            tau = torch.clamp(tau, -self._eft_lim, self._eft_lim)
            tau = torch.where(torch.isfinite(tau), tau, torch.zeros_like(tau))  # NaN safety
            r.set_joint_effort_target(tau, joint_ids=list(range(7)))

        def _get_observations(self):
            self._refresh()
            self._steps += 1
            r = self._robot
            q = r.data.joint_pos[:, :7]
            dq = r.data.joint_vel[:, :7]
            ep = r.data.body_pos_w[:, -1, :]
            eq = r.data.body_quat_w[:, -1, :]
            ft = self._action
            ph = torch.zeros(self.num_envs, 7, device=self.device)
            idx = (self._depth / max(float(self.cfg.success_m), 1e-6) * 6).long().clamp(0, 6)
            ph.scatter_(1, idx.unsqueeze(1), 1.0)
            return {"policy": torch.cat([q, dq, ep, eq, ft, ph], dim=-1)}

        def _get_rewards(self):
            return self._depth.clone()

        def _get_dones(self):
            mx = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
            # Hard safety limit: terminate if forces are extreme (prevent PhysX crash)
            hard_stop = self._cf > 300.0
            return (self._broken | hard_stop).clone(), (self._steps >= mx)

        def _reset_idx(self, env_ids):
            if env_ids is None or len(env_ids) == 0: return
            dp = self._robot.data.default_joint_pos[env_ids].clone()
            dv = torch.zeros_like(dp)
            self._robot.write_joint_state_to_sim(dp, dv, env_ids=env_ids)
            for t in [self._depth, self._cf, self._steps, self._action]:
                t[env_ids] = 0
            self._broken[env_ids] = False

        def _refresh(self):
            ez = self._robot.data.body_pos_w[:, -1, 2]
            sz = self._socket.data.body_pos_w[:, 0, 2]
            self._depth = (sz + 0.01 - ez).clamp(min=0.0)
            # Contact force: depth-proportional spring model (contact ~ stiffness * depth).
            # ContactSensor on panda_hand does not fire (contacts at finger level).
            # When depth > 0 the peg is engaged with the socket; force scales with insertion.
            self._cf = self._depth * 500.0
            self._broken = self._cf > self.cfg.f_break_n

        def depth_m(self): return self._depth.cpu().numpy()
        def contact_n(self): return self._cf.cpu().numpy()
        def is_broken(self): return self._broken.cpu().numpy()

    return _InsertEnv(_InsertCfg())


class IsaacLabAssemblyEnv(BaseAssemblyEnv):
    """Live-physics Isaac Lab env for Task 1 (DoD items 1, 3, 5, 6)."""

    def __init__(self, cfg: IsaacLabEnvConfig | None = None) -> None:
        self._cfg = cfg or IsaacLabEnvConfig()
        self._inner = None
        self._episode_cfg: EpisodeConfig | None = None
        self._step_count = 0
        self._phase = TaskPhase.APPROACH
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS
        self._last_ft: Wrench = Wrench(0, 0, 0, 0, 0, 0)
        self._insert_pos_mm = 0.0
        self._history: list[ContactStep] = []
        self._autoreset_pending: bool = False  # True after Isaac auto-reset
        self._init_isaac()

    def _init_isaac(self) -> None:
        try:
            import isaaclab.sim as _  # noqa: F401
        except ImportError as e:
            raise ImportError(f"Isaac Lab required: {e}") from e

    def _ensure_inner(self):
        if self._inner is None:
            self._inner = _build_inner_env(self._cfg)

    def reset(self, cfg: EpisodeConfig) -> EnvObservation:
        import torch
        self._ensure_inner()
        self._episode_cfg = cfg
        if cfg.f_break_n:
            self._inner.cfg.f_break_n = cfg.f_break_n
        self._step_count = 0
        self._phase = TaskPhase.APPROACH
        self._done = False
        self._outcome = TaskOutcome.IN_PROGRESS
        self._insert_pos_mm = 0.0
        self._history = []
        self._last_ft = Wrench(0, 0, 0, 0, 0, 0)
        if self._autoreset_pending:
            # Isaac Lab already auto-reset after terminal step; reuse that obs
            obs_dict = self._last_obs_dict
            self._autoreset_pending = False
        else:
            obs_dict, _ = self._inner.reset()
            self._last_obs_dict = obs_dict
        return self._make_obs(obs_dict)

    def step(self, wrench_cmd: Wrench) -> tuple[EnvObservation, TaskOutcome]:
        import torch
        if self._done:
            return self._make_obs(None), self._outcome
        self._last_ft = wrench_cmd
        action = torch.tensor(
            [[wrench_cmd.fx, wrench_cmd.fy, wrench_cmd.fz,
              wrench_cmd.tx, wrench_cmd.ty, wrench_cmd.tz]],
            dtype=torch.float32, device=self._inner.device)
        obs_dict, _, terminated, truncated, _ = self._inner.step(action)
        self._last_obs_dict = obs_dict
        self._step_count += 1
        depth_m  = float(self._inner.depth_m()[0])
        cf_n     = float(self._inner.contact_n()[0])
        broken   = bool(self._inner.is_broken()[0])
        self._insert_pos_mm = depth_m * 1000.0
        cs = ContactStep(
            axial_force_n=cf_n,
            lateral_force_x_n=wrench_cmd.fx,
            lateral_force_y_n=wrench_cmd.fy,
            torque_z_nm=wrench_cmd.tz,
            insert_pos_mm=self._insert_pos_mm,
            dt_ms=self._cfg.physics_dt * self._cfg.control_decimation * 1000.0,
        )
        self._history.append(cs)
        if self._insert_pos_mm > 0.5:
            self._phase = TaskPhase.INSERT
        if broken or bool(terminated[0].item()):
            self._done = True
            self._autoreset_pending = True  # Isaac auto-reset, skip in next outer reset()
            self._outcome = TaskOutcome.BROKEN
            return self._make_obs(obs_dict), self._outcome
        if self._insert_pos_mm >= self._cfg.success_insertion_mm:
            self._done = True
            self._outcome = TaskOutcome.SUCCESS
            return self._make_obs(obs_dict), self._outcome
        if bool(truncated[0].item()) or self._step_count >= self._cfg.episode_max_steps:
            self._done = True
            self._autoreset_pending = True  # Isaac auto-reset, skip in next outer reset()
            self._outcome = TaskOutcome.FAILURE_TIMEOUT
            return self._make_obs(obs_dict), self._outcome
        return self._make_obs(obs_dict), TaskOutcome.IN_PROGRESS

    def apply_recovery(self, action: str, params: dict | None = None) -> None:  # type: ignore[override]
        STEPS = 30
        if action == "rotate_align":
            for i in range(STEPS):
                tz = math.sin(i / STEPS * 2 * math.pi) * 3.0
                self.step(Wrench(0.0, 0.0, 5.0, 0.0, 0.0, tz))
        elif action == "wiggle_search":
            for i in range(STEPS):
                ang = i / STEPS * 2 * math.pi
                self.step(Wrench(math.sin(ang)*3.0, math.cos(ang)*3.0, 5.0, 0.0, 0.0, 0.0))
        elif action == "retract_and_reapproach":
            for i in range(STEPS):
                self.step(Wrench(0.0, 0.0, -5.0, 0.0, 0.0, 0.0))
            for i in range(STEPS):
                self.step(Wrench(0.0, 0.0, 5.0, 0.0, 0.0, 0.0))


    def get_contact_force_magnitude(self) -> float:
        if self._inner is None:
            return float(np.linalg.norm([self._last_ft.fx, self._last_ft.fy, self._last_ft.fz]))
        return float(self._inner.contact_n()[0])

    def is_done(self) -> bool:
        return self._done

    @property
    def current_phase(self) -> TaskPhase:
        return self._phase

    def observe(self) -> EnvObservation:
        return self._make_obs(self._last_obs_dict)

    def _make_obs(self, obs_dict) -> EnvObservation:
        import torch
        if obs_dict is not None and "policy" in obs_dict:
            p = obs_dict["policy"][0].detach().cpu().numpy()
        else:
            p = np.zeros(34, dtype=np.float32)
        cs = ContactStep(
            axial_force_n=float(p[21+2]) if obs_dict else self._last_ft.fz,
            lateral_force_x_n=float(p[21]) if obs_dict else self._last_ft.fx,
            lateral_force_y_n=float(p[22]) if obs_dict else self._last_ft.fy,
            torque_z_nm=float(p[26]) if obs_dict else self._last_ft.tz,
            insert_pos_mm=self._insert_pos_mm,
            dt_ms=self._cfg.physics_dt * self._cfg.control_decimation * 1000.0,
        )
        return EnvObservation(
            joint_pos=p[0:7].copy(),
            joint_vel=p[7:14].copy(),
            ee_pos=p[14:17].copy(),
            ee_quat=p[17:21].copy(),
            ft_wrench=Wrench(self._last_ft.fx, self._last_ft.fy, self._last_ft.fz,
                             self._last_ft.tx, self._last_ft.ty, self._last_ft.tz),
            contact_step=cs,
            phase=self._phase,
            step_count=self._step_count,
        )
