"""FrankaPlaceEnv - Task 3 fragile place/stack DirectRLEnv (real physics).

The Franka lowers its hand onto a kinematic rack/surface; a ContactSensor
measures the contact force. A hidden per-instance F_break (evaluator only)
defines breakage. Success = gentle contact (force in a small band) sustained
for a short window. Over-pressing (force > F_break) breaks the part -- "press
harder" is maximally destructive here.

F_cmd is sampled per env and exposed to the FORGE policy via FiLM (env.f_cmd_norm);
the reward penalizes exceeding F_cmd, while breakage uses the hidden F_break.
Control is 7-dim delta joint-position (no IK dependency -> robust to boot).
Obs(34) = joint_pos(7)+joint_vel(7)+ee_pos(3)+ee_quat(4)+ft_wrench(6)+phase(7).
"""
from __future__ import annotations
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_mul, quat_inv


@configclass
class PlaceEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=4)
    episode_length_s: float = 6.0
    decimation: int = 2
    observation_space: int = 34
    action_space: int = 7
    state_space: int = 0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
    action_scale: float = 0.08
    ee_action_scale: float = 0.02
    place_z: float = 0.665      # hand-z at contact with rack (env-rel)
    lam: float = 0.025         # FORGE action clip (m): max force = kp*lam
    act_range: float = 0.05    # relative action offset from fixed part (m)
    gripper: str = "franka_panda"  # or "robotiq_2f140" (modelled via contact compliance)
    rack_top_z: float = 0.665
    f_cmd_lo: float = 10.0
    f_cmd_hi: float = 100.0
    settle_force_n: float = 3.0
    settle_steps: int = 3
    contact_eps_n: float = 0.2


class FrankaPlaceEnv(DirectRLEnv):
    cfg: PlaceEnvCfg
    NUM_PHASES = 7

    def __init__(self, cfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode=render_mode, **kw)
        self._actions = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_quat_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._ee_quat_des[:, 0] = 1.0
        self._osc_init = False
        self._kp_pos = 1200.0
        self._kd_pos = 80.0   # ~2*sqrt(kp)
        self._kp_ori = 50.0
        self._kd_ori = 14.0   # ~2*sqrt(kp_ori)
        self._kd_joint = 1.0
        self._eff_lim = torch.tensor([87., 87., 87., 87., 12., 12., 12.], device=self.device)
        self._ee_set_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._set_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._home_ee_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._offset_scale = 0.25
        _gmap = {"franka_panda": (4000.0, 900.0), "robotiq_2f140": (1800.0, 1400.0)}
        self._grip_ks, self._grip_kd = _gmap.get(self.cfg.gripper, (4000.0, 500.0))
        self._cf_filt = torch.zeros(self.num_envs, device=self.device)
        self._cf_alpha = 0.15
        self._az_filt = torch.full((self.num_envs,), -1.0, device=self.device)
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._f_cmd = torch.zeros(self.num_envs, device=self.device)
        self._f_break = torch.zeros(self.num_envs, device=self.device)
        self._settle_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._broke = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ee_idx = self._robot.find_bodies("panda_hand")[0][0]
        self._arm_ids = self._robot.find_joints(["panda_joint[1-7]"])[0]
        self._jt_target = self._robot.data.default_joint_pos[:, self._arm_ids].clone()
        self._jacobi_idx = self._ee_idx - 1  # fixed base: jacobian body index
        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
            num_envs=self.num_envs, device=self.device)
        self._sample_episode(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):
        _gmap = {"franka_panda": (4000.0, 900.0), "robotiq_2f140": (1800.0, 1400.0)}
        self._grip_ks, self._grip_kd = _gmap.get(self.cfg.gripper, (4000.0, 500.0))
        try:
            from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
            robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
            robot_cfg.spawn.usd_path = "/workspace/assets/franka/panda_instanceable.usd"
            robot_cfg.spawn.activate_contact_sensors = True
        except Exception as e:
            raise RuntimeError(f"FRANKA_PANDA_CFG unavailable: {e}")
        for _an in ("panda_shoulder", "panda_forearm"):
            robot_cfg.actuators[_an].stiffness = 0.0
            robot_cfg.actuators[_an].damping = 0.0
        _jp = dict(robot_cfg.init_state.joint_pos)
        _jp["panda_joint2"] = -0.73
        _jp["panda_joint4"] = -2.46
        _jp["panda_joint6"] = 2.85
        robot_cfg.init_state.joint_pos = _jp
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot

        table = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Table",
            spawn=sim_utils.CuboidCfg(size=(0.9, 0.6, 0.4),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.2))))
        self.scene.rigid_objects["table"] = table

        rack = RigidObject(RigidObjectCfg(
            prim_path="/World/envs/env_.*/Rack",
            spawn=sim_utils.CuboidCfg(size=(0.12, 0.12, 0.05), activate_contact_sensors=True, physics_material=sim_utils.RigidBodyMaterialCfg(compliant_contact_stiffness=self._grip_ks, compliant_contact_damping=self._grip_kd),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.38, 0.0, 0.610))))
        self.scene.rigid_objects["rack"] = rack

        self._contact = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_(hand|leftfinger|rightfinger)",
            update_period=0.0, history_length=1, debug_vis=False,
            filter_prim_paths_expr=["/World/envs/env_.*/Rack"]))
        self.scene.sensors["contact"] = self._contact

        sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
        self.scene.clone_environments(copy_from_source=False)

    def _sample_episode(self, ids):
        n = len(ids)
        frag = torch.rand(n, device=self.device) < 0.5
        fb = torch.where(frag,
                         torch.rand(n, device=self.device) * 20 + 15,
                         torch.rand(n, device=self.device) * 110 + 150)
        self._f_break[ids] = fb
        # F_cmd (the budget shown to the policy) is a SAFE, feasible fraction of
        # F_break, as the LLM supervisor sets F_max below the breaking force.
        self._f_cmd[ids] = fb * (0.5 + 0.35 * torch.rand(n, device=self.device))

    def _raw_contact_force(self):
        nf = getattr(self._contact.data, "net_forces_w", None)
        if nf is None:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.norm(nf, dim=-1).sum(dim=1)

    def _contact_force(self):
        # low-pass filtered contact force (ignores impulsive chatter)
        return self._cf_filt

    def f_cmd_norm(self):
        return (self._f_cmd / 120.0).unsqueeze(-1)

    def _pre_physics_step(self, actions):
        self._actions = actions.clamp(-1, 1)

    def _apply_action(self):
        # FORGE controller: target = fixed place pose + relative action, clipped to
        # +/- lam of the EE so contact force is bounded by kp*lam (gentle by design).
        r = self._robot
        self._cf_filt = (1.0 - self._cf_alpha) * self._cf_filt + self._cf_alpha * self._raw_contact_force()
        jac = r.root_physx_view.get_jacobians()[:, self._jacobi_idx, :, :7]
        ee_pos_w = r.data.body_pos_w[:, self._ee_idx]
        ee_quat_w = r.data.body_quat_w[:, self._ee_idx]
        if not self._osc_init:
            self._ee_quat_des = ee_quat_w.clone()
            self._osc_init = True
        ee_lin_v = r.data.body_lin_vel_w[:, self._ee_idx]
        ee_ang_v = r.data.body_ang_vel_w[:, self._ee_idx]
        p_fixed = self.scene.env_origins + torch.tensor([0.38, 0.0, self.cfg.place_z], device=self.device)
        a = self._actions[:, :3] * self.cfg.act_range
        delta = (p_fixed + a - ee_pos_w).clamp(-self.cfg.lam, self.cfg.lam)
        force = self._kp_pos * delta - self._kd_pos * ee_lin_v
        q_err = quat_mul(self._ee_quat_des, quat_inv(ee_quat_w))
        ang_err = 2.0 * torch.sign(q_err[:, 0:1]) * q_err[:, 1:4]
        moment = self._kp_ori * ang_err - self._kd_ori * ee_ang_v
        wrench = torch.cat([force, moment], dim=-1).unsqueeze(-1)
        tau = (jac.transpose(1, 2) @ wrench).squeeze(-1)
        tau = tau - self._kd_joint * r.data.joint_vel[:, :7]
        grav = r.root_physx_view.get_gravity_compensation_forces()[:, self._arm_ids]
        tau = tau + grav
        tau = torch.clamp(tau, -self._eff_lim, self._eff_lim)
        r.set_joint_effort_target(tau, joint_ids=self._arm_ids)

    def _get_observations(self):
        jp = self._robot.data.joint_pos[:, :7]
        jv = self._robot.data.joint_vel[:, :7]
        ee_p = self._robot.data.body_pos_w[:, self._ee_idx, :] - self.scene.env_origins
        ee_q = self._robot.data.body_quat_w[:, self._ee_idx, :]
        cf = self._contact_force().unsqueeze(-1)
        ft = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), cf,
                        torch.zeros(self.num_envs, 3, device=self.device)], dim=-1)
        ph = torch.zeros(self.num_envs, self.NUM_PHASES, device=self.device)
        ph.scatter_(1, self._phase.unsqueeze(1), 1.0)
        return {"policy": torch.cat([jp, jv, ee_p, ee_q, ft, ph], dim=-1)}

    def _get_rewards(self):
        ee_z = self._robot.data.body_pos_w[:, self._ee_idx, 2] - self.scene.env_origins[:, 2]
        cf = self._contact_force()
        height_err = (ee_z - self.cfg.rack_top_z).clamp(min=0.0)
        in_contact = cf > self.cfg.contact_eps_n
        good = in_contact & (cf < self._f_cmd)
        firm = torch.clamp(cf / self._f_cmd, 0.0, 1.0)
        r = -0.3 * height_err
        r = r + good.float() * (1.0 + 2.0 * firm)
        r = r + good.float() * (self._settle_ctr.float() / self.cfg.settle_steps)
        r = r - 2.0 * (cf - self._f_cmd).clamp(min=0.0) / self._f_cmd  # FORGE excessive-force penalty
        r = r - self._broke.float() * 10.0 + self._succeeded.float() * 10.0
        return r

    def _get_dones(self):
        cf = self._contact_force()
        grace = self.episode_length_buf > 3  # ignore reset contact transient
        self._broke = (cf > self._f_break) & grace
        gentle = (cf > self.cfg.contact_eps_n) & (cf < self._f_cmd)
        self._settle_ctr = torch.where(gentle, self._settle_ctr + 1, torch.zeros_like(self._settle_ctr))
        self._succeeded = self._settle_ctr >= self.cfg.settle_steps
        terminated = self._broke | self._succeeded
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["succ_mask"] = self._succeeded.clone()
        self.extras["brk_mask"] = self._broke.clone()
        self.extras["n_succ"] = float(self._succeeded.sum().item())
        self.extras["n_brk"] = float(self._broke.sum().item())
        return terminated, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        jp = self._robot.data.default_joint_pos[env_ids]
        jv = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])
        self._robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
        self._phase[env_ids] = 0
        self._settle_ctr[env_ids] = 0
        self._broke[env_ids] = False
        self._succeeded[env_ids] = False
        self._set_reset[env_ids] = True
        self._cf_filt[env_ids] = 0.0
        self._az_filt[env_ids] = -1.0
        self._jt_target[env_ids] = self._robot.data.default_joint_pos[env_ids][:, self._arm_ids]
        self._sample_episode(env_ids)


import gymnasium as gym
gym.register(id="FORGE-Place-v0",
             entry_point="forge_plus.isaac_place_env:FrankaPlaceEnv",
             kwargs={"cfg": PlaceEnvCfg()})
