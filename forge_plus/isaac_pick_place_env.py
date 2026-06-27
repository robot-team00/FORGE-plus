#!/usr/bin/env python3
"""FrankaPickPlaceEnv — FORGE-plus Task 3 pick-and-place environment.

Picks a fragile object from a low TABLE (~0.4 m) and places it on an
overhead RACK (~1.1 m), simulating kitchen cabinet loading.

Objects (Issue #28 with F_break):
  glass_bowl   22±4 N    ceramic_plate 26±5 N
  metal_plate  180±25 N  sturdy_mug    160±20 N

LLM (Ollama llama3.1:8b) sets F_max < F_break per object; cached in
llm/budget_cache.json.  Force is monitored at GRASP and PLACE_DESCEND.

Obs  (34): joint_pos(7)|joint_vel(7)|ee_pos(3)|ee_quat(4)|ft_wrench(6)|phase_onehot(7)
Action (7): delta_ee_pos(3)|delta_ee_quat(4)  — delta EE pose, OSC control
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np
import torch

# ── Isaac Lab imports (fall back to mock on CPU-only nodes) ─────────────────
try:
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils.math import (
        quat_mul, quat_inv, matrix_from_quat, quat_from_angle_axis,
        subtract_frame_transforms, quat_apply_inverse,
    )
    from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
    import isaaclab.sim as sim_utils
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False

    def configclass(cls):  # noqa: F811
        import dataclasses
        return dataclasses.dataclass(cls)

import gymnasium as gym


# ─────────────────────────────────────────────────────────────────────────────
# Phase definition (7 phases → phase_onehot(7) in obs)
# ─────────────────────────────────────────────────────────────────────────────
class PickPlacePhase(IntEnum):
    PRE_GRASP     = 0   # hover above object on table
    DESCEND       = 1   # lower EE to object
    GRASP         = 2   # close gripper; pick-force monitored
    LIFT          = 3   # lift object to transport altitude
    TRANSPORT     = 4   # sweep horizontally toward overhead rack
    PLACE_DESCEND = 5   # lower EE to rack surface; place-force monitored
    RELEASE       = 6   # open gripper → success


NUM_PHASES = 7   # == len(PickPlacePhase), must equal phase_onehot width


# ─────────────────────────────────────────────────────────────────────────────
# Fragile object registry  (FORGE-plus Issue #28)
# ─────────────────────────────────────────────────────────────────────────────
FRAGILE_OBJECTS: dict[str, dict] = {
    #              F_break dist                grasp
    "glass_bowl":    {"f_mean": 22.0,  "f_std":  4.0, "f_min":  12.0, "grasp_mm":  80.0},
    "ceramic_plate": {"f_mean": 26.0,  "f_std":  5.0, "f_min":  14.0, "grasp_mm": 120.0},
    "metal_plate":   {"f_mean": 180.0, "f_std": 25.0, "f_min": 120.0, "grasp_mm": 120.0},
    "sturdy_mug":    {"f_mean": 160.0, "f_std": 20.0, "f_min": 110.0, "grasp_mm":  95.0},
}
OBJ_KEYS  = list(FRAGILE_OBJECTS.keys())   # fixed ordering for tensor indexing
N_OBJ_CLS = len(OBJ_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# LLM budget helper  (Ollama llama3.1:8b, result cached on disk)
# ─────────────────────────────────────────────────────────────────────────────
_BUDGET_CACHE = "/workspace/FORGE-plus_task3/llm/budget_cache.json"

_MAT_HINT = {
    "glass_bowl":    "borosilicate glass",
    "ceramic_plate": "stoneware ceramic",
    "metal_plate":   "aluminium alloy",
    "sturdy_mug":    "thick-wall stoneware",
}


def _llm_query_budget(obj_key: str) -> float:
    """Ask Ollama for a safe F_max (N) for this object. Returns conservative fallback on failure."""
    cfg  = FRAGILE_OBJECTS[obj_key]
    name = obj_key.replace("_", " ")
    mat  = _MAT_HINT.get(obj_key, "unknown material")
    prompt = (
        f"Robot safety task. A Franka Panda arm picks up a {name} made of {mat}. "
        f"The object breaks if the contact force exceeds roughly {cfg['f_mean']:.0f} N. "
        f"What is the maximum safe contact force in Newtons the robot should apply "
        f"when grasping or placing this object? "
        f"Reply with one integer only."
    )
    try:
        env_copy = os.environ.copy()
        env_copy["OLLAMA_HOME"] = "/workspace/ollama_models"
        res = subprocess.run(
            ["/workspace/bin/ollama", "run", "llama3.1:8b", prompt],
            capture_output=True, text=True, timeout=90, env=env_copy,
        )
        m = re.search(r"\d+", res.stdout.strip())
        if m:
            v = float(m.group())
            v = max(v, cfg["f_min"] * 0.35)          # safety floor
            v = min(v, cfg["f_mean"] * 0.70)         # safety ceiling
            return v
    except Exception as exc:
        print(f"[LLM] Ollama failed for {obj_key}: {exc}", flush=True)
    return cfg["f_mean"] * 0.40   # conservative fallback: 40 % of mean


def _load_or_query_budgets() -> dict[str, float]:
    """Load cached LLM budgets, querying Ollama for any missing entries."""
    os.makedirs(os.path.dirname(_BUDGET_CACHE), exist_ok=True)
    cache: dict[str, float] = {}
    if os.path.exists(_BUDGET_CACHE):
        try:
            cache = json.load(open(_BUDGET_CACHE))
        except Exception:
            cache = {}

    updated = False
    for key in OBJ_KEYS:
        if key not in cache:
            print(f"[LLM] querying F_max for {key} ...", flush=True)
            cache[key] = _llm_query_budget(key)
            print(
                f"[LLM]   {key}: F_max = {cache[key]:.1f} N "
                f"(F_break ~ {FRAGILE_OBJECTS[key]['f_mean']:.0f} N)",
                flush=True,
            )
            updated = True

    if updated:
        with open(_BUDGET_CACHE, "w") as fh:
            json.dump(cache, fh, indent=2)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Environment configuration
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class PickPlaceEnvCfg(DirectRLEnvCfg if ISAAC_AVAILABLE else object):  # type: ignore[misc]
    # Simulation
    sim: object = SimulationCfg(dt=1.0 / 120.0, render_interval=4) if ISAAC_AVAILABLE else None
    episode_length_s: float = 30.0   # 600 steps: the lam-clipped OSC moves slowly
                                     # (~80 steps/phase), so the full 7-phase pick-
                                     # transport-place needs ~400+ steps. At 12 s
                                     # (240 steps) episodes timed out at LIFT before
                                     # ever reaching RELEASE -> no success signal.
    decimation: int = 2    # matches the proven FrankaPlaceEnv. (decimation=8 only
                           # made things worse because the Jacobian bug made the
                           # controller unstable; with the fix, 2 is smooth.)

    # Spaces — must match ForceConditionedPolicy defaults (obs_dim=34, act_dim=7)
    observation_space: int = 34
    action_space: int      = 7
    state_space: int       = 0

    # Parallel envs
    scene: object = (
        InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)
        if ISAAC_AVAILABLE else None
    )

    # OSC controller (inherited from FrankaPlaceEnv tuning)
    action_scale: float    = 0.08
    ee_action_scale: float = 0.02
    lam: float             = 0.025   # FORGE action clip (m): |delta| <= lam
    act_range: float       = 0.05    # position action scale

    # ── Spatial layout: low TABLE → overhead RACK ─────────────────────────
    # All z values are in the env-local frame (world_z = env_origin_z + env_z).
    table_top_z:   float = 0.40   # table surface height
    obj_rest_z:    float = 0.44   # object resting z on table (centre + half-height)
    pre_grasp_z:   float = 0.60   # hover above object before descend
    transport_z:   float = 0.72   # safe altitude for horizontal swing. At radius
                                  # ~0.46 the Franka tops out near z≈0.72, so the old
                                  # 0.80 was unreachable — the hand stalled at ~0.70
                                  # and rammed the object into the (too-tall) rack.
    # Place target: the EE hand height at which the cup's base meets the shelf top.
    # Decoupled from the shelf geometry below. shelf_top 0.50 + cup half 0.045 + the
    # hand->grasp-point offset (~0.067) ≈ 0.61 -> the cup is SET DOWN gently on top.
    place_ee_z:    float = 0.53    # EE hand height for the PLACE_DESCEND waypoint. For "insert"
                                   # this drives the bottle base down to the cell floor (~0.40).
    place_settle_tol: float = 0.02  # |cup_base - shelf_top| to count the cup as placed
    mug_grip_z: float = 0.12   # height up the mug (from its base origin) where the gripper grips
    # Hold the hand TOP-DOWN (approach axis straight down) so a neck-gripped tall object hangs
    # VERTICAL and can be PLACED STANDING. FORGE holds the part at a fixed correct roll/pitch;
    # here that's upright. Needs higher OSC orientation stiffness (the default 40 barely tracks).
    grasp_topdown: bool = False  # if True, command a fixed top-down ee_quat_des at warmup
    # Placement strategy:
    #   "throw_upright" = contact-then-verticalize (preserved old demo: rights the bottle against
    #                     the shelf via a stiffness ramp — looks like flicking it upright).
    #   "extrinsic"     = LEARNED extrinsic dexterity: the policy uses a wrist-pitch action +
    #                     the counter contact to PIVOT the bottle upright, gently. No rigid hold.
    #   "insert" = wine-cellar PEG-IN-HOLE: lower the bottle into a rack cell; the cell walls
    #              align it (contact-rich, FORGE force-guided) and hold it upright. Success = the
    #              base reaches the cell floor (inserted to depth) gently.
    place_strategy: str = "insert"
    rack_z: float = 0.38           # wine-rack origin z; cell top at +0.12 (=0.50), floor at +0.02 (=0.40)
    cell_floor_z: float = 0.40     # bottle base target depth (cell floor) for a full insertion
    insert_depth_tol: float = 0.03 # |base - cell_floor| below this counts as inserted
    ori_k_insert: float = 110.0    # firmish grip during insertion (wide cell -> no bind) so the
                                   # bottle goes in ~upright instead of leaning over
    ori_k_descend: float = 40.0    # throw_upright: orientation stiffness during descent
    ori_k_vertical: float = 200.0  # throw_upright: ramped-to stiffness on shelf-contact
    vert_ramp_steps: int = 18      # throw_upright: ramp length
    # extrinsic dexterity (validated mechanic): firm grip during carry to limit the lean, then a
    # COMPLIANT grip once the base is on the shelf so the policy can roll the bottle upright about
    # the contact with LATERAL position moves (the wrist-pitch route slammed it, so it's dropped).
    ori_k_carry: float = 120.0     # firm-ish during LIFT/TRANSPORT/descent (keeps the lean small)
    ori_k_extrinsic: float = 12.0  # COMPLIANT once planted, so it can pivot on the contact
    compliant_band: float = 0.05   # base within shelf_top + this -> start ramping to compliant
    comp_ramp: int = 15            # steps to SMOOTHLY ramp firm->compliant (avoids a lurch/force spike)
    lam_place: float = 0.012       # extrinsic: SLOW per-step EE motion once planted, so the roll-up
                                   # stays gentle (low contact force) regardless of action direction
    upright_cos_tol: float = 0.985 # cos(tilt) above this (~10 deg) counts as upright
    require_upright: bool = True   # curriculum: stage-A places (False), stage-B adds the pivot (True)
    grasp_com_drop: float = 0.0     # seat the cup centre this far BELOW the pads so the
                                    # grip is above the COM (pendulum-stable, stays upright)
    shelf_top_z:   float = 0.50   # reachable-from-above shelf/counter surface; = wine-cell top

    # Horizontal offsets from robot base in env frame
    table_x: float = 0.45   # table centre x
    rack_x:  float = 0.45   # rack/cell x (in front, well within reach for base-aimed insertion)
    rack_y:  float = 0.12   # rack/cell y (modest lateral offset; reachable so the base centers)

    # Gripper
    gripper: str   = "franka_panda"

    # Place-only mode: start each episode already holding the object at transport
    # altitude and only do TRANSPORT -> PLACE_DESCEND -> RELEASE (gentle place).
    # Skips the pick/grasp/lift phases (matches the proven FrankaPlaceEnv task).
    place_only: bool = True

    # Force thresholds / settle criteria
    f_cmd_lo:         float = 6.0
    f_cmd_hi:         float = 120.0
    contact_eps_n:    float = 0.15   # min force to count as contact (N)
    grasp_force_n:    float = 0.5    # min force to confirm grasp
    place_force_n:    float = 0.3    # min force to confirm rack contact (a gentle
                                     # place makes light contact; 1.5 N was above
                                     # where the cautious policy settles -> hover)
    settle_steps:     int   = 15   # gentle-contact steps at RELEASE for a placed
                                   # success. The policy reliably reaches RELEASE
                                   # with a gentle, under-budget place (settle_ctr
                                   # hits 1 on every env); episodes cycle before a
                                   # 2nd step accrues, so 1 step = a placed success.
    warmup_substeps:  int   = 10   # ~5 env steps (decimation 2) to seat the grip; was
                                   # 25 (~12 steps), which ate most of the short demo and
                                   # put the grip-settle transient late in the descent.

    # Phase advance tolerances. 0.08 (was 0.05) because the Jacobian-transpose OSC
    # has a steady-state error of a few cm near reach-limited waypoints; 0.05 was
    # too tight and stalled the LIFT/TRANSPORT advance.
    reach_tol: float = 0.08   # EE proximity to phase waypoint (m)
    place_reach_tol: float = 0.03   # TIGHT proximity for PLACE_DESCEND. The loose 0.08
                                    # made the cup "close" to the place target from the
                                    # start (it begins only ~7 cm above), so the grip-
                                    # settle force blip at warmup-end spuriously completed
                                    # the place ~7 cm ABOVE the shelf. 0.03 forces a real
                                    # descent-to-contact before RELEASE.


# ─────────────────────────────────────────────────────────────────────────────
# Mock / CPU backend  (no Isaac Sim needed — for unit tests and dev loops)
# ─────────────────────────────────────────────────────────────────────────────
class MockPickPlaceEnv(gym.Env):
    """Lightweight CPU mock of FrankaPickPlaceEnv.

    No Isaac Sim required.  Joint kinematics are approximated; contact forces
    are simulated from the phase.  Use this to verify obs shapes, reward logic,
    and training loop plumbing before GPU runs.
    """

    metadata = {"render_modes": []}

    def __init__(self, num_envs: int = 4, device: str = "cpu"):
        super().__init__()
        self.num_envs = num_envs
        self.device   = torch.device(device)
        self.cfg      = PickPlaceEnvCfg()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (34,), np.float32)
        self.action_space      = gym.spaces.Box(-1.0, 1.0, (7,), np.float32)

        N, d = num_envs, self.device
        # Franka home configuration (7 joints)
        self._q0 = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], device=d)
        self._jp  = self._q0.unsqueeze(0).repeat(N, 1)
        self._jv  = torch.zeros(N, 7, device=d)
        self._ee_pos  = torch.zeros(N, 3, device=d)
        self._ee_quat = torch.zeros(N, 4, device=d)
        self._ee_quat[:, 0] = 1.0   # w=1 (identity)
        self._ft      = torch.zeros(N, 6, device=d)   # 6-D force-torque wrench
        self._phase   = torch.zeros(N, dtype=torch.long, device=d)
        self._step    = torch.zeros(N, dtype=torch.long, device=d)
        self._f_break = torch.zeros(N, device=d)
        self._f_cmd   = torch.zeros(N, device=d)
        self._broke   = torch.zeros(N, dtype=torch.bool, device=d)
        self._done    = torch.zeros(N, dtype=torch.bool, device=d)
        self._extras: dict = {}
        self._max_steps    = int(self.cfg.episode_length_s * 20)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def f_cmd_norm(self) -> torch.Tensor:
        return (self._f_cmd / 120.0).unsqueeze(-1)

    def _sample_episode(self, ids) -> None:
        n = len(ids)
        cls = torch.randint(0, N_OBJ_CLS, (n,), device=self.device)
        fm  = torch.tensor([FRAGILE_OBJECTS[k]["f_mean"] for k in OBJ_KEYS], device=self.device)
        fs  = torch.tensor([FRAGILE_OBJECTS[k]["f_std"]  for k in OBJ_KEYS], device=self.device)
        fn  = torch.tensor([FRAGILE_OBJECTS[k]["f_min"]  for k in OBJ_KEYS], device=self.device)
        fb  = fm[cls] + fs[cls] * torch.randn(n, device=self.device)
        fb  = torch.maximum(fb, fn[cls])
        self._f_break[ids] = fb
        self._f_cmd[ids]   = fm[cls] * 0.45   # 45 % of mean: safe budget

    def _phase_target(self) -> torch.Tensor:
        N, d, c = self.num_envs, self.device, self.cfg
        tgt = torch.zeros(N, 3, device=d)
        waypoints = [
            (c.table_x, 0.0,      c.pre_grasp_z),    # PRE_GRASP
            (c.table_x, 0.0,      c.obj_rest_z),      # DESCEND
            (c.table_x, 0.0,      c.obj_rest_z),      # GRASP
            (c.rack_x,  c.rack_y, c.transport_z),     # LIFT
            (c.rack_x,  c.rack_y, c.transport_z),     # TRANSPORT
            (c.rack_x,  c.rack_y, c.place_ee_z),       # PLACE_DESCEND
            (c.rack_x,  c.rack_y, c.transport_z),      # RELEASE: lift the hand AWAY
        ]                                              # (leaves the cup resting on top)
        for ph, (x, y, z) in enumerate(waypoints):
            m = self._phase == ph
            tgt[m, 0] = x; tgt[m, 1] = y; tgt[m, 2] = z
        return tgt

    def _get_obs(self) -> dict:
        ph = torch.zeros(self.num_envs, NUM_PHASES, device=self.device)
        ph.scatter_(1, self._phase.unsqueeze(1), 1.0)
        obs = torch.cat([self._jp, self._jv, self._ee_pos, self._ee_quat, self._ft, ph], dim=-1)
        return {"policy": obs}

    # ── Gym interface ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        N, d = self.num_envs, self.device
        self._phase[:]  = 0
        self._step[:]   = 0
        self._broke[:]  = False
        self._done[:]   = False
        self._jp[:]     = self._q0.unsqueeze(0)
        self._jv[:]     = 0.0
        self._ft[:]     = 0.0
        self._ee_pos[:, 0] = self.cfg.table_x
        self._ee_pos[:, 1] = 0.0
        self._ee_pos[:, 2] = self.cfg.pre_grasp_z
        self._ee_quat[:, 0] = 1.0; self._ee_quat[:, 1:] = 0.0
        self._sample_episode(list(range(N)))
        return self._get_obs(), {}

    def step(self, action):
        N, d = self.num_envs, self.device
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=d)
        action = action.to(d).clamp(-1, 1)
        self._step += 1

        # Move EE toward phase target
        tgt   = self._phase_target()
        delta = (tgt - self._ee_pos).clamp(-self.cfg.lam, self.cfg.lam)
        self._ee_pos = self._ee_pos + delta * 0.4   # partial step
        self._jp += torch.randn_like(self._jp) * 0.002
        self._jv  = torch.randn_like(self._jv) * 0.01

        # Simulate contact force
        cf = torch.zeros(N, device=d)
        for ph_idx, scale in [(PickPlacePhase.GRASP, 0.60), (PickPlacePhase.PLACE_DESCEND, 0.55)]:
            m   = self._phase == int(ph_idx)
            cf  = torch.where(m,
                self._f_cmd * scale + torch.randn(N, device=d) * 1.5,
                cf)
        self._ft[:, 0] = cf      # Fx (dominant contact component)
        self._ft[:, 1:] = 0.0

        # Break check (only at force-active phases)
        force_active = (self._phase == PickPlacePhase.GRASP) |                        (self._phase == PickPlacePhase.PLACE_DESCEND)
        grace = self._step > 5
        self._broke = self._broke | (grace & (cf > self._f_break) & force_active)

        # Phase advance when close to target — EXCEPT PLACE_DESCEND, which advances to
        # RELEASE only once the cup actually RESTS on the shelf (contact > place_force_n).
        # Releasing on mere proximity (reach_tol 8 cm) dropped the cup from ~8 cm up and
        # spiked the contact force; gating release on real contact makes it a gentle
        # set-down. The descent force is still breakage-monitored, so the policy must
        # come down gently to contact (FORGE soft-place behaviour).
        dist  = (self._ee_pos - tgt).norm(dim=-1)
        close = dist < self.cfg.reach_tol
        place_ph   = self._phase == int(PickPlacePhase.PLACE_DESCEND)
        contact_ok = cf > self.cfg.place_force_n
        advance = torch.where(place_ph, close & contact_ok, close)
        next_ph = (self._phase + 1).clamp(max=NUM_PHASES - 1)
        self._phase = torch.where(advance & ~self._broke, next_ph, self._phase)

        # Done / success
        succeeded  = self._phase >= int(PickPlacePhase.RELEASE)
        terminated = self._broke | succeeded
        truncated  = self._step >= self._max_steps

        # Reward
        rew  = -0.2 * dist
        rew  = rew + self._phase.float() * 0.5
        rew  = rew + succeeded.float() * 10.0
        rew  = rew - self._broke.float() * 10.0
        excess = (cf - self._f_cmd).clamp(min=0.0) / self._f_cmd.clamp(min=1.0)
        rew  = rew - 2.0 * excess

        self._extras = {
            "n_succ": float(succeeded.sum().item()),
            "n_brk":  float(self._broke.sum().item()),
        }
        return self._get_obs(), rew, terminated, truncated, self._extras

    def render(self): pass
    def close(self): pass


# ─────────────────────────────────────────────────────────────────────────────
# Full Isaac Lab environment
# ─────────────────────────────────────────────────────────────────────────────
if ISAAC_AVAILABLE:

    class FrankaPickPlaceEnv(DirectRLEnv):
        """FORGE-plus Task 3: Franka Panda picks fragile object from table, places on overhead rack.

        Force budget (F_cmd) < F_break is enforced at both pick (GRASP) and
        place (PLACE_DESCEND) phases — the core FORGE safety-margin demonstration.

        Scene layout
        ------------
        * Robot base at env origin.
        * TABLE: flat cuboid at x=+0.45, z_surface=0.40 m.
        * RACK:  thin bar at  x=+0.35, y=+0.30, z=1.10 m (overhead cabinet sim).
        * Height delta rack - table = 0.70 m (meaningful vertical reach requirement).
        """

        cfg: PickPlaceEnvCfg
        NUM_PHASES = NUM_PHASES

        def __init__(self, cfg: PickPlaceEnvCfg, render_mode=None, **kw):
            super().__init__(cfg, render_mode=render_mode, **kw)
            N, d = self.num_envs, self.device

            # Proper Isaac Lab Operational-Space Controller: task-space impedance
            # WITH inertia decoupling (Lambda = (J M^-1 J^T)^-1) and null-space
            # control for the redundant 7th DOF. The previous hand-rolled
            # Jacobian-transpose controller had neither -> poorly-conditioned task
            # dynamics + an undamped elbow/wrist -> the jitter/wobble.
            osc_cfg = OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                # variable_kp: orientation stiffness is set PER STEP via the command, so we can
                # descend with low orientation stiffness (place works) and ramp it up on
                # shelf-contact to RIGHT the bottle about the contact pivot (force-guided settle).
                impedance_mode="variable_kp",
                motion_stiffness_limits_task=(5.0, 600.0),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
                motion_stiffness_task=[400.0, 400.0, 400.0, 40.0, 40.0, 40.0],  # initial; overridden per step
                motion_damping_ratio_task=1.0,                  # critically damped
                motion_control_axes_task=[1, 1, 1, 1, 1, 1],
                nullspace_control="position",
                nullspace_stiffness=15.0,
                nullspace_damping_ratio=1.0,
            )
            self._osc = OperationalSpaceController(osc_cfg, num_envs=N, device=d)
            self._joint_centers = None   # null-space posture target (set lazily)
            self._eff_lim  = torch.tensor([87., 87., 87., 87., 12., 12., 12.], device=d)
            _gmap = {"franka_panda": (80.0, 10.0), "robotiq_2f140": (60.0, 10.0)}
            self._grip_ks, self._grip_kd = _gmap.get(self.cfg.gripper, (80.0, 10.0))

            # Indices (resolved after scene build)
            self._arm_ids:  list[int] = list(range(7))  # arm joint indices 0-6
            self._ee_idx:   int = -1  # resolved lazily in _reset_idx
            self._lf_idx:   int = -1  # panda_leftfinger  (held-object grasp centre)
            self._rf_idx:   int = -1  # panda_rightfinger
            self._osc_init: bool = False

            # Actions / EE state
            self._actions     = torch.zeros(N, 7, device=d)
            self._prev_actions = torch.zeros(N, 7, device=d)  # for action-rate smoothness
            self._ee_quat_des = torch.zeros(N, 4, device=d)
            self._ee_quat_des[:, 0] = 1.0
            self._gripper_cmd = torch.ones(N, device=d)   # +1 = open, -1 = closed

            # Contact force (low-pass filter)
            self._cf_filt  = torch.zeros(N, device=d)
            self._cf_alpha = 0.15

            # Episode state
            self._phase      = torch.zeros(N, dtype=torch.long, device=d)
            self._phase_ctr  = torch.zeros(N, dtype=torch.long, device=d)
            self._settle_ctr = torch.zeros(N, dtype=torch.long, device=d)
            # Contact-then-verticalize: counts steps the base has been on the shelf during
            # PLACE_DESCEND; ramps the OSC orientation stiffness up to right the bottle.
            self._vert_ctr   = torch.zeros(N, dtype=torch.long, device=d)
            self._best_tilt  = torch.full((N,), 3.1416, device=d)   # extrinsic: best (min) tilt this place
            self._f_cmd      = torch.zeros(N, device=d)
            self._f_break    = torch.zeros(N, device=d)
            self._broke      = torch.zeros(N, dtype=torch.bool, device=d)
            self._succeeded  = torch.zeros(N, dtype=torch.bool, device=d)
            self._advanced   = torch.zeros(N, dtype=torch.bool, device=d)  # advanced a phase this step
            self._set_reset  = torch.zeros(N, dtype=torch.bool, device=d)
            self._warmup     = torch.zeros(N, dtype=torch.long, device=d)
            # Distance from the hand origin to the grasp point (between the fingertip
            # pads) along the hand's local +z. Per-env so it can be swept/calibrated;
            # 0.067 seats the block centrally between the pads (calibrated).
            self._grasp_tcp_d = torch.full((N,), 0.067, device=d)
            # Finger half-opening to rest the pads at during warmup: block half-width
            # (0.020) minus 1 mm so the pads sit just at the surface (clean seat, no
            # deep penetration), then the PD grip takes over and holds by friction.
            self._grasp_seat_w = 0.006
            # PD gains for the position-controlled grip (effort = k·(target-pos) - kd·vel).
            # k·overlap sets the squeeze: 1500 N/m · 0.010 m = 15 N grip (friction
            # 1.6·15 = 24 N >> the 0.5 N block weight), penetration sub-mm under rigid contact.
            self._grip_pos_ks = 1500.0
            self._grip_pos_kd = 40.0
            self._az_filt    = torch.full((N,), -1.0, device=d)
            self._jt_target  = torch.zeros(N, 7, device=d)
            self._extras: dict = {}

            # Object registry tensors
            self._n_obj_cls = N_OBJ_CLS
            self._obj_fmean = torch.tensor([FRAGILE_OBJECTS[k]["f_mean"] for k in OBJ_KEYS], device=d)
            self._obj_fstd  = torch.tensor([FRAGILE_OBJECTS[k]["f_std"]  for k in OBJ_KEYS], device=d)
            self._obj_fmin  = torch.tensor([FRAGILE_OBJECTS[k]["f_min"]  for k in OBJ_KEYS], device=d)
            self._obj_cls   = torch.zeros(N, dtype=torch.long, device=d)

            # LLM budgets (Ollama query, cached)
            budgets = _load_or_query_budgets()
            raw     = [budgets.get(k, FRAGILE_OBJECTS[k]["f_mean"] * 0.4) for k in OBJ_KEYS]
            self._obj_budget = torch.tensor(raw, device=d)
            print(
                "[PickPlace] LLM F_max budgets: "
                + ", ".join(f"{k}={raw[i]:.1f}N" for i, k in enumerate(OBJ_KEYS)),
                flush=True,
            )

        # ── Scene ─────────────────────────────────────────────────────────────
        def _setup_scene(self) -> None:
            from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

            # Compliant-contact gains (also set in __init__, but _setup_scene runs
            # first via super().__init__ — set here so they exist for the spawn cfgs).
            _gmap = {"franka_panda": (80.0, 10.0), "robotiq_2f140": (60.0, 10.0)}
            self._grip_ks, self._grip_kd = _gmap.get(self.cfg.gripper, (80.0, 10.0))
            # Dedicated (softer) compliant stiffness for the fragile contact surfaces.
            # The gripper stiffness (4 kN/m) is near-rigid: any touch spikes contact
            # force to ~150 N, far above the fragile budgets (9-72 N), so the policy
            # cannot modulate gently. ~1.2 kN/m leaves room to press softly into the
            # 2-72 N range the task needs.
            self._surf_ks, self._surf_kd = 1200.0, 120.0

            # Robot
            robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
            robot_cfg.spawn.usd_path = "/workspace/assets/franka/panda_instanceable.usd"
            robot_cfg.spawn.activate_contact_sensors = True
            # Disable the joint PD controllers on the proximal joints so the OSC
            # (Jacobian-transpose effort targets) actually moves the arm instead of
            # being overpowered back to the default pose. Mirrors FrankaPlaceEnv.
            # Effort control for the OSC: zero the joint-position stiffness so the
            # actuator PD doesn't fight the OSC, but KEEP joint-velocity damping --
            # without it the torque-controlled joints buzz at high frequency (the
            # vibration). The damping opposes joint velocity and kills the buzz.
            for _an in ("panda_shoulder", "panda_forearm"):
                robot_cfg.actuators[_an].stiffness = 0.0
                robot_cfg.actuators[_an].damping = 80.0
            _jp = dict(robot_cfg.init_state.joint_pos)
            _jp["panda_joint2"] = -0.73
            _jp["panda_joint4"] = -2.46
            _jp["panda_joint6"] = 2.85
            robot_cfg.init_state.joint_pos = _jp
            self._robot = Articulation(robot_cfg)

            # Table (low flat surface)
            table_h = self.cfg.table_top_z
            table_spawn = sim_utils.UsdFileCfg(
                usd_path="{NVIDIA_NUCLEUS_DIR}/Assets/Props/Furniture/table/table.usd",
                scale=(0.6, 0.6, table_h),
            ) if False else sim_utils.CuboidCfg(size=(0.6, 0.6, table_h), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), collision_props=sim_utils.CollisionPropertiesCfg())  # use primitive cuboid
            self._table = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/envs/env_.*/Table",
                    spawn=table_spawn,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(self.cfg.table_x, 0.0, table_h / 2.0),
                    ),
                )
            )

            # Graspable fragile object on the table (kinematic + compliant contact).
            # Pressing it during DESCEND/GRASP yields a controllable contact force
            # gated by F_cmd/F_break — mirrors the proven FrankaPlaceEnv physics so
            # the grasp-force gate (cf > grasp_force_n) is actually reachable.
            # Graspable object: a small light DYNAMIC block with high friction so the
            # closed gripper actually holds it (real physics grasp). Repositioned
            # into the gripper at reset/warmup; released onto the rack at RELEASE.
            self._obj = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/envs/env_.*/Object",
                    spawn=sim_utils.UsdFileCfg(
                        # A REAL fragile kitchen object: the LIBERO wine_bottle (glass,
                        # MIT-licensed mesh). The Franka grips its narrow ~1.6 cm NECK
                        # (scale 0.5) near the top; the heavy glass body hangs BELOW the
                        # grip, so it's pendulum-stable and self-rights upright -> a real
                        # FRICTION grasp holds it through the carry (a round mug body can't).
                        # Origin is at the base; the seat offsets by mug_grip_z to grip the neck.
                        usd_path="/workspace/assets/libero/wine_bottle/wine_bottle_rigid.usd",
                        scale=(0.5, 0.5, 0.5),
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.30),
                        # DYNAMIC — real physics: held by the friction grip, carried, released.
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False, disable_gravity=False,
                            max_depenetration_velocity=1.0,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(self.cfg.rack_x, self.cfg.rack_y, self.cfg.transport_z),
                    ),
                )
            )

            # Wine-cellar RACK: a grid of vertical cells the bottle is INSERTED into (peg-in-hole).
            # Origin = center-cell xy; cell bottom at +base_t, cell top at +0.12 in the rack frame.
            # Placed so the center cell top sits at shelf_top_z (0.50) -> cell bottom at rack_z+0.02.
            self._rack = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/envs/env_.*/Rack",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/workspace/assets/libero/wine_rack/wine_rack.usd",
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(self.cfg.rack_x, self.cfg.rack_y, self.cfg.rack_z),
                    ),
                )
            )

            # Contact sensor on hand + fingers, filtered to the object and rack so
            # net_forces_w reports the grasp/place contact force (panda_hand alone
            # never touches either surface → was reading ~0 N).
            self._contact_sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_(hand|leftfinger|rightfinger)",
                    update_period=0.0,
                    history_length=1,
                    track_air_time=False,
                    filter_prim_paths_expr=[
                        "/World/envs/env_.*/Object",
                        "/World/envs/env_.*/Rack",
                    ],
                )
            )
            self._contact_data = self._contact_sensor

            # Surface contact sensor — net force applied *to* the Object/Rack by the
            # gripper. This is the physically meaningful fragile-contact force and is
            # immune to the gripper's finger-on-finger self-contact (which pollutes
            # the robot sensor's net_forces_w when the gripper closes on air).
            self._surf_sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/envs/env_.*/(Object|Rack)",
                    update_period=0.0,
                    history_length=1,
                    track_air_time=False,
                )
            )

            # Register with scene
            self.scene.articulations["robot"]  = self._robot
            self.scene.rigid_objects["table"]  = self._table
            self.scene.rigid_objects["object"] = self._obj
            self.scene.rigid_objects["rack"]   = self._rack
            self.scene.sensors["contact"]      = self._contact_sensor
            self.scene.sensors["surf"]         = self._surf_sensor
            self.scene.clone_environments(copy_from_source=False)

            # Cache joint / body indices
            self._arm_ids = list(range(7))
            self._ee_idx  = -1  # resolved lazily in _reset_idx (data not ready at setup time)

        # ── Episode sampling ──────────────────────────────────────────────────
        def _sample_episode(self, ids) -> None:
            n   = len(ids)
            cls = torch.randint(0, self._n_obj_cls, (n,), device=self.device)
            fb  = (self._obj_fmean[cls]
                   + self._obj_fstd[cls] * torch.randn(n, device=self.device))
            fb  = torch.maximum(fb, self._obj_fmin[cls])
            self._f_break[ids] = fb
            self._f_cmd[ids]   = self._obj_budget[cls]
            self._obj_cls[ids] = cls

        # ── Contact force helpers ─────────────────────────────────────────────
        def _raw_contact_force(self) -> torch.Tensor:
            # Net force applied TO the Object/Rack by the gripper, read from the
            # surface sensor's `.data.net_forces_w` (shape (N, bodies, 3)). This is
            # the fragile-contact force and excludes gripper self-contact.
            sdata = getattr(self._surf_sensor, "data", None)
            snf = getattr(sdata, "net_forces_w", None) if sdata is not None else None
            if snf is not None and snf.numel() > 0:
                return torch.norm(snf, dim=-1).sum(dim=1)
            # Fallback: robot sensor filtered to Object+Rack.
            data = getattr(self._contact_sensor, "data", None)
            fm = getattr(data, "force_matrix_w", None) if data is not None else None
            if fm is not None and fm.numel() > 0:
                return torch.norm(fm, dim=-1).sum(dim=(1, 2))
            return torch.zeros(self.num_envs, device=self.device)

        def _contact_force(self) -> torch.Tensor:
            """Low-pass filtered scalar contact force."""
            return self._cf_filt

        def _contact_wrench_6d(self) -> torch.Tensor:
            """6D force-torque for observation (ft_wrench slot)."""
            data = getattr(self._contact_sensor, "data", None)
            nf = getattr(data, "net_forces_w", None) if data is not None else None
            if nf is None:
                return torch.zeros(self.num_envs, 6, device=self.device)
            flat = nf.reshape(self.num_envs, -1)
            if flat.shape[1] >= 6:
                return flat[:, :6]
            pad = torch.zeros(self.num_envs, 6 - flat.shape[1], device=self.device)
            return torch.cat([flat, pad], dim=1)

        def f_cmd_norm(self) -> torch.Tensor:
            """Normalised F_cmd in [0, 1] for policy conditioning."""
            return (self._f_cmd / 120.0).unsqueeze(-1)

        # ── Phase waypoint helpers ────────────────────────────────────────────
        def _phase_waypoint_world(self) -> torch.Tensor:
            """Return the fixed OSC target in world frame for the current phase of each env."""
            N, d, c = self.num_envs, self.device, self.cfg
            orig = self.scene.env_origins   # (N, 3)
            wp   = orig.clone()
            waypoints = [
                (c.table_x, 0.0,      c.pre_grasp_z),
                (c.table_x, 0.0,      c.obj_rest_z),
                (c.table_x, 0.0,      c.obj_rest_z),
                (c.table_x, 0.0,      c.transport_z),   # LIFT: straight up at table
                (c.rack_x,  c.rack_y, c.transport_z),   # TRANSPORT: over to the shelf
                (c.rack_x,  c.rack_y, c.place_ee_z),     # PLACE_DESCEND: set down on top
                (c.rack_x,  c.rack_y, c.transport_z),    # RELEASE: lift hand away
            ]
            for ph, (x, y, z) in enumerate(waypoints):
                m = self._phase == ph
                wp[m, 0] = orig[m, 0] + x
                wp[m, 1] = orig[m, 1] + y
                wp[m, 2] = orig[m, 2] + z
            return wp

        def _phase_target_z_local(self) -> torch.Tensor:
            """Return per-env target z in env-local frame (for reward shaping)."""
            c  = self.cfg
            zs = torch.tensor([
                c.pre_grasp_z, c.obj_rest_z, c.obj_rest_z,
                c.transport_z, c.transport_z, c.place_ee_z, c.place_ee_z,
            ], device=self.device)
            return zs[self._phase.clamp(max=NUM_PHASES - 1)]

        # ── Physics step ──────────────────────────────────────────────────────
        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            self._actions = actions.clamp(-1, 1)

        def _apply_action(self) -> None:
            """FORGE OSC controller: target = phase waypoint + policy delta."""
            r    = self._robot
            _raw = self._raw_contact_force()
            _live = (self._warmup == 0)
            self._cf_filt = torch.where(
                _live,
                (1.0 - self._cf_alpha) * self._cf_filt + self._cf_alpha * _raw,
                torch.zeros_like(self._cf_filt),
            )
            self._warmup = (self._warmup - 1).clamp(min=0)

            # ── Seat the dynamic object in the gripper during warmup ───────────
            # While the grip settles, snap the object to the grasp centre (finger
            # midpoint) with zero velocity. After warmup the closed gripper holds it
            # by friction (no more snapping -> it's a real physics grasp).
            warm = self._warmup > 0
            # Seat the vessel in the gripper during warmup; AFTER warmup the closed gripper
            # holds it by REAL FRICTION (no teleport) — a faithful physics grasp. Grip a
            # thin stem/neck near the top so the COM hangs below (pendulum -> self-rights
            # upright and resists tilting through the carry).
            carry = warm
            if carry.any():
                R_ee = matrix_from_quat(r.data.body_quat_w[:, self._ee_idx])      # (N,3,3)
                off  = torch.zeros(self.num_envs, 3, device=self.device)
                off[:, 2] = self._grasp_tcp_d
                grasp_c = r.data.body_pos_w[:, self._ee_idx] + torch.bmm(R_ee, off.unsqueeze(-1)).squeeze(-1)
                pose = self._obj.data.root_pose_w.clone()
                # Hold the vessel UPRIGHT (must stand). Its origin is at the BASE, so drop it
                # by mug_grip_z so the gripper sits on the upper body.
                pose[carry, 0:3] = grasp_c[carry]
                pose[carry, 2] = grasp_c[carry, 2] - self.cfg.mug_grip_z
                pose[carry, 3] = 1.0; pose[carry, 4:7] = 0.0
                self._obj.write_root_pose_to_sim(pose)
                vel = self._obj.data.root_vel_w.clone()
                vel[carry] = 0.0
                self._obj.write_root_velocity_to_sim(vel)

            # ── End-effector state in the robot base (root) frame ─────────────
            root_pos_w, root_quat_w = r.data.root_pos_w, r.data.root_quat_w
            ee_pos_w  = r.data.body_pos_w[:, self._ee_idx]
            ee_quat_w = r.data.body_quat_w[:, self._ee_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
            ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
            lin_b = quat_apply_inverse(root_quat_w, r.data.body_lin_vel_w[:, self._ee_idx] - r.data.root_lin_vel_w)
            ang_b = quat_apply_inverse(root_quat_w, r.data.body_ang_vel_w[:, self._ee_idx] - r.data.root_ang_vel_w)
            ee_vel_b = torch.cat([lin_b, ang_b], dim=-1)

            # Jacobian (fixed base -> body i at index i-1), rotated to base frame
            jac_b = r.root_physx_view.get_jacobians()[:, self._ee_idx - 1, :, :7].clone()
            Rb = matrix_from_quat(quat_inv(root_quat_w))
            jac_b[:, :3, :] = torch.bmm(Rb, jac_b[:, :3, :])
            jac_b[:, 3:, :] = torch.bmm(Rb, jac_b[:, 3:, :])

            mm   = r.root_physx_view.get_generalized_mass_matrices()[:, self._arm_ids, :][:, :, self._arm_ids]
            try:
                grav = r.root_physx_view.get_gravity_compensation_forces()[:, self._arm_ids]
            except Exception:
                grav = r.root_physx_view.get_generalized_gravity_forces()[:, self._arm_ids]
            if self._joint_centers is None:
                self._joint_centers = r.data.joint_pos[:, self._arm_ids].clone()  # nominal posture

            if not self._osc_init:
                self._ee_quat_natural = ee_quat_w.clone()   # the as-grasped hand orientation
                if self.cfg.grasp_topdown:
                    self._ee_quat_des = torch.tensor(
                        [0.0, 1.0, 0.0, 0.0], device=self.device
                    ).expand(self.num_envs, 4).clone()
                else:
                    self._ee_quat_des = ee_quat_w.clone()   # hold the initial grasp orientation
                self._osc_init = True

            # ── Placement strategy: orientation stiffness + ee_quat_des ───────────
            lam_eff = torch.full((self.num_envs,), self.cfg.lam, device=self.device)  # per-step EE motion cap
            if self.cfg.place_strategy == "throw_upright":
                # Contact-then-verticalize (preserved old demo): ramp orientation stiffness once
                # the base is on the shelf and command top-down -> rights the bottle against the
                # contact (looks like flicking it upright).
                base_z = self._obj.data.root_pose_w[:, 2] - self.scene.env_origins[:, 2]
                on_shelf_now = (self._phase == int(PickPlacePhase.PLACE_DESCEND)) & \
                               (base_z < self.cfg.shelf_top_z + self.cfg.place_settle_tol)
                self._vert_ctr = torch.where(on_shelf_now, self._vert_ctr + 1, self._vert_ctr)
                frac  = (self._vert_ctr.float() / float(max(1, self.cfg.vert_ramp_steps))).clamp(max=1.0)
                ori_k = self.cfg.ori_k_descend + frac * (self.cfg.ori_k_vertical - self.cfg.ori_k_descend)
                vmask = frac > 0.0
                if vmask.any():
                    self._ee_quat_des[vmask] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
            elif self.cfg.place_strategy == "extrinsic":  # LEARNED extrinsic dexterity (validated mechanic)
                base_z = self._obj.data.root_pose_w[:, 2] - self.scene.env_origins[:, 2]
                planted = (self._phase >= int(PickPlacePhase.PLACE_DESCEND)) & \
                          (base_z < self.cfg.shelf_top_z + self.cfg.compliant_band)
                self._vert_ctr = torch.where(planted, self._vert_ctr + 1, self._vert_ctr)
                frac = (self._vert_ctr.float() / float(max(1, self.cfg.comp_ramp))).clamp(max=1.0)
                ori_k = self.cfg.ori_k_carry - frac * (self.cfg.ori_k_carry - self.cfg.ori_k_extrinsic)
                self._ee_quat_des = self._ee_quat_natural
                lam_eff = torch.where(planted, torch.full_like(base_z, self.cfg.lam_place),
                                      torch.full_like(base_z, self.cfg.lam))
            else:  # "insert": wine-cellar PEG-IN-HOLE
                # FIRM grip during carry (stable, base stays aligned -> base-aim can center it over
                # the cell); soften to a MODERATE grip near the cell so it can align as it goes in.
                # Slow the motion near the cell so the contact-rich insertion stays gentle.
                base_z = self._obj.data.root_pose_w[:, 2] - self.scene.env_origins[:, 2]
                near = (self._phase >= int(PickPlacePhase.PLACE_DESCEND)) & \
                       (base_z < self.cfg.cell_floor_z + 0.10)
                ori_k = torch.where(near, torch.full_like(base_z, self.cfg.ori_k_insert),
                                    torch.full_like(base_z, self.cfg.ori_k_carry))
                self._ee_quat_des = self._ee_quat_natural
                lam_eff = torch.where(near, torch.full_like(base_z, self.cfg.lam_place),
                                      torch.full_like(base_z, self.cfg.lam))

            # Target pose = lam-rate-limited step toward the phase waypoint + policy
            # delta (keeps FORGE's bounded per-step motion), in base frame.
            p_fixed  = self._phase_waypoint_world()
            a        = self._actions[:, :3] * self.cfg.act_range
            # "insert": aim the BOTTLE BASE at the cell, not the gripper — the ~12 deg lean offsets
            # the base ~6 cm from the EE, so move the EE target by that offset to center the base.
            if self.cfg.place_strategy == "insert":
                off_xy = self._obj.data.root_pose_w[:, :2] - ee_pos_w[:, :2]   # base-to-EE horizontal offset
                approach = (self._phase >= int(PickPlacePhase.TRANSPORT)).unsqueeze(-1)
                p_fixed = torch.cat([p_fixed[:, :2] - approach.float() * off_xy, p_fixed[:, 2:3]], dim=-1)
            _lam = lam_eff.unsqueeze(-1)
            target_w = ee_pos_w + (p_fixed + a - ee_pos_w).clamp(min=-_lam, max=_lam)
            tgt_pos_b, tgt_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_w, self._ee_quat_des)
            _k400 = torch.full_like(ori_k, 400.0)
            stiffness = torch.stack([_k400, _k400, _k400, ori_k, ori_k, ori_k], dim=-1)   # (N, 6)
            command  = torch.cat([tgt_pos_b, tgt_quat_b, stiffness], dim=-1)  # variable_kp: pose(7)+stiffness(6)

            # Operational-space control: inertia-decoupled task impedance + gravity
            # comp + null-space posture control (no more hand-rolled Jacobian-T).
            self._osc.set_command(command=command, current_ee_pose_b=ee_pose_b)
            jt = self._osc.compute(
                jacobian_b=jac_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                mass_matrix=mm,
                gravity=grav,
                current_joint_pos=r.data.joint_pos[:, self._arm_ids],
                current_joint_vel=r.data.joint_vel[:, self._arm_ids],
                nullspace_joint_pos_target=self._joint_centers,
            )
            jt = jt.clamp(-self._eff_lim, self._eff_lim)

            # Gripper closes while carrying (TRANSPORT/PLACE_DESCEND) and at GRASP;
            # opens at RELEASE to drop the block onto the rack. The block is DYNAMIC,
            # so the closed gripper genuinely holds it by friction.
            close_mask = (
                (self._phase == int(PickPlacePhase.GRASP)) |
                (self._phase == int(PickPlacePhase.LIFT)) |
                (self._phase == int(PickPlacePhase.TRANSPORT)) |
                (self._phase == int(PickPlacePhase.PLACE_DESCEND))
            )
            self._gripper_cmd = torch.where(close_mask.float().bool(), -torch.ones_like(self._gripper_cmd), torch.ones_like(self._gripper_cmd))
            fvel = r.data.joint_vel[:, 7:9]
            fpos = r.data.joint_pos[:, 7:9]
            # Position-controlled grip (PD effort). The closed target sits INSIDE the
            # block half-width, so the pads press the faces with a bounded squeeze
            # force (k·overlap) and rest AT the surface — no slam-through. (A constant
            # effort grip has no position feedback and over-penetrated to ~5 mm finger
            # width; PD self-limits.) Open target retracts the pads to drop the block.
            _d = self.device
            target = torch.where(
                close_mask,
                torch.full((self.num_envs,), 0.002, device=_d),   # squeeze the thin (~1.6 cm) bottle neck
                torch.full((self.num_envs,), 0.040, device=_d),   # fully open  -> release
            )
            # During warmup rest exactly at the surface while the block is teleported in.
            target = torch.where(warm, torch.full((self.num_envs,), self._grasp_seat_w, device=_d), target)
            gforce = self._grip_pos_ks * (target.unsqueeze(1) - fpos) - self._grip_pos_kd * fvel
            r.set_joint_effort_target(torch.cat([jt, gforce], dim=-1))

        # ── Observations ──────────────────────────────────────────────────────
        def _get_observations(self) -> dict:
            r    = self._robot
            jp   = r.data.joint_pos[:, self._arm_ids]           # (N, 7)
            jv   = r.data.joint_vel[:, self._arm_ids]           # (N, 7)
            ee_p = r.data.body_pos_w[:, self._ee_idx] - self.scene.env_origins  # (N, 3)
            ee_q = r.data.body_quat_w[:, self._ee_idx]         # (N, 4)
            ft   = self._contact_wrench_6d()                    # (N, 6)
            # Object orientation: its local +z (up) axis in world. The policy needs this to
            # sense the bottle's tilt/lean so it can pivot it upright (extrinsic dexterity).
            obj_up = torch.bmm(matrix_from_quat(self._obj.data.root_pose_w[:, 3:7]),
                               torch.tensor([0.,0.,1.], device=self.device).view(1,3,1).expand(self.num_envs,3,1)).squeeze(-1)  # (N,3)
            ph   = torch.zeros(self.num_envs, self.NUM_PHASES, device=self.device)
            ph.scatter_(1, self._phase.unsqueeze(1), 1.0)       # (N, 7) one-hot
            # cat → (N, 7+7+3+4+6+3+7) = (N, 37)
            return {"policy": torch.cat([jp, jv, ee_p, ee_q, ft, obj_up, ph], dim=-1)}

        # ── Rewards ───────────────────────────────────────────────────────────
        def _get_rewards(self) -> torch.Tensor:
            c    = self.cfg
            ee_z = (self._robot.data.body_pos_w[:, self._ee_idx, 2]
                    - self.scene.env_origins[:, 2])
            cf   = self._contact_force()

            # Height shaping toward phase target z
            tgt_z    = self._phase_target_z_local()
            h_err    = (ee_z - tgt_z).abs()
            r = -0.3 * h_err
            # Strong descent incentive at PLACE_DESCEND: the policy was holding the
            # arm up near transport height (its up-action cancelling the OSC's
            # downward drive) and never contacting the rack. Heavily reward being at
            # the rack height so it descends and makes gentle contact.
            place_m = (self._phase == int(PickPlacePhase.PLACE_DESCEND)).float()
            r = r - 2.5 * place_m * h_err

            # Phase-progress: one-time +2 when a phase is COMPLETED (drives progression).
            r = r + self._advanced.float() * 2.0
            # NOTE: removed the per-step `+0.1*phase_index` living bonus and the per-step
            # force-in-window bonus. Both pay positive reward every step while merely
            # NEAR the target, so their horizon-integral exceeded the one-time success
            # bonus -> the optimal policy was to HOVER at PLACE_DESCEND forever and never
            # commit (succ collapsed while return rose). FORGE pays no positive per-step
            # bonus; only penalties + a success cliff (Ng 1999 PBRS / reward-hacking).

            # Small per-step time cost so waiting is strictly worse than committing.
            r = r - 0.02

            # Uprightness shaping (extrinsic) — reward PROGRESS toward upright (a NEW best-so-far
            # tilt), never penalising tilt increases. A symmetric reward punished the random
            # exploration (which mostly increases tilt) and taught the policy to FREEZE; rewarding
            # only new lows lets it explore the roll-up freely, and "new low" can't be farmed.
            if self.cfg.place_strategy == "extrinsic":
                up_z = matrix_from_quat(self._obj.data.root_pose_w[:, 3:7])[:, 2, 2].clamp(-1.0, 1.0)
                tilt = torch.acos(up_z)                                    # radians from vertical
                in_pivot = self._phase >= int(PickPlacePhase.PLACE_DESCEND)
                first = in_pivot & (self._best_tilt >= 3.0)                # first pivot step this episode
                self._best_tilt = torch.where(first, tilt, self._best_tilt)
                improve = ((self._best_tilt - tilt).clamp(min=0.0)) * in_pivot.float()
                r = r + 25.0 * improve                                    # reward each new low (progress)
                self._best_tilt = torch.where(in_pivot, torch.minimum(self._best_tilt, tilt), self._best_tilt)

            # Force-excess penalty (FORGE: penalise exceeding F_cmd). Bounded so an
            # over-press guides the policy down instead of catastrophically swamping
            # the progression rewards (unbounded -2*excess hit ~-33/step at cf~150 vs
            # a 9 N budget, which taught the policy to avoid contact entirely).
            excess = ((cf - self._f_cmd).clamp(min=0.0) / self._f_cmd.clamp(min=1.0)).clamp(max=3.0)
            r = r - 0.5 * excess   # softer deterrent so the policy commits to contact

            # Force-MARGIN penalty (extrinsic): the roll-up presses the base on the shelf, so keep
            # a safety margin BELOW the break force — penalise the contact once it exceeds half of
            # f_break, well before it actually breaks. Teaches a gentle roll-up instead of slamming.
            if self.cfg.place_strategy == "extrinsic":
                margin = ((cf - 0.5 * self._f_break).clamp(min=0.0) / self._f_break.clamp(min=1.0)).clamp(max=2.0)
                r = r - 1.0 * margin

            # Smoothness regularizers (reduce the visibly shaky motion): penalise
            # fast joint motion and rapid action changes so the policy learns to
            # move calmly. Damping the OSC gains alone did not help — the jitter is
            # policy-commanded.
            # Moderate smoothness penalty: enough to calm the motion without making
            # the deterministic policy too sluggish to complete the transport
            # (0.12/0.45 over-damped it -> stuck at TRANSPORT). Residual high-freq
            # ripple is low-passed at render time.
            jvel = self._robot.data.joint_vel[:, self._arm_ids].abs().mean(dim=-1)
            r = r - 0.05 * jvel
            arate = (self._actions - self._prev_actions).abs().mean(dim=-1)
            r = r - 0.22 * arate
            self._prev_actions = self._actions.clone()

            # Terminal rewards: strongly reward completing the gentle place so the
            # policy stops hovering; keep a real but not-terrifying breakage penalty
            # (contact is gentle -> breakage stays ~0 anyway).
            r = r + self._succeeded.float() * 20.0
            r = r - self._broke.float() * 6.0
            return r

        # ── Dones ─────────────────────────────────────────────────────────────
        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            c  = self.cfg
            cf = self._contact_force()

            grace = (self.episode_length_buf > 3) & (self._warmup == 0)

            # Break check ONLY at force-monitored phases
            force_active = (
                (self._phase == int(PickPlacePhase.GRASP)) |
                (self._phase == int(PickPlacePhase.PLACE_DESCEND))
            )
            self._broke = (cf > self._f_break) & grace & force_active

            # Phase advance
            ee_z  = (self._robot.data.body_pos_w[:, self._ee_idx, 2]
                     - self.scene.env_origins[:, 2])
            tgt_z = self._phase_target_z_local()
            # Tight tolerance at PLACE_DESCEND so the cup must actually reach the shelf
            # (not "succeed" 7 cm up off a grip-settle blip); loose elsewhere.
            tol = torch.where(
                self._phase == int(PickPlacePhase.PLACE_DESCEND),
                torch.full_like(ee_z, c.place_reach_tol),
                torch.full_like(ee_z, c.reach_tol),
            )
            close = (ee_z - tgt_z).abs() < tol
            # Contact phases press a compliant surface and cannot reach the
            # sub-surface waypoint z (the surface stops the fingers above it), so
            # "reached" is satisfied by making contact rather than z-proximity.
            # Without this, DESCEND never completes and the grasp/place phases are
            # never entered → 0 breakage and 0 success forever.
            contact_phase = (
                (self._phase == int(PickPlacePhase.DESCEND)) |
                (self._phase == int(PickPlacePhase.GRASP))
            )
            close = close | (contact_phase & (cf > c.contact_eps_n))
            # Sweep phases (LIFT, TRANSPORT) must REACH the waypoint x,y — not just match
            # the transport height — otherwise the arm skips them instantly (both are at
            # transport_z) and never visibly moves across. Block their advance while the
            # EE is still far from the waypoint in the horizontal plane.
            ee_xy = self._robot.data.body_pos_w[:, self._ee_idx, :2]
            wp_xy = self._phase_waypoint_world()[:, :2]
            xy_far = (ee_xy - wp_xy).norm(dim=-1) > 0.06
            sweep_ph = (self._phase == int(PickPlacePhase.LIFT)) | (self._phase == int(PickPlacePhase.TRANSPORT))
            close = close & ~(sweep_ph & xy_far)
            # PLACE_DESCEND completes GEOMETRICALLY: the hand/finger contact sensor
            # cannot see the cup resting on the shelf (the gentle cup-shelf force nets
            # ~0 through the friction grip), so gate the place on the cup's BASE
            # actually reaching the shelf surface, measured from the cup's real pose.
            cup_z     = self._obj.data.root_pose_w[:, 2] - self.scene.env_origins[:, 2]
            cup_bot   = cup_z   # mug origin is at its BASE, so root z == base height
            if c.place_strategy == "insert":
                # INSERTED = base down at the cell FLOOR (not the cell top) AND inside the cell xy.
                base_xy = self._obj.data.root_pose_w[:, :2] - self.scene.env_origins[:, :2]
                in_cell = ((base_xy[:, 0] - c.rack_x).abs() < 0.04) & ((base_xy[:, 1] - c.rack_y).abs() < 0.04)
                on_shelf = ((cup_bot - c.cell_floor_z).abs() < c.insert_depth_tol) & in_cell
            else:
                on_shelf = (cup_bot - c.shelf_top_z).abs() < c.place_settle_tol
            place_dsc = self._phase == int(PickPlacePhase.PLACE_DESCEND)
            # Bottle upright? (cos of tilt = local-z's world z). For "extrinsic", the place only
            # completes (advances to RELEASE) once the base is on the shelf AND it is UPRIGHT, so
            # the policy must pivot it up before letting go; "throw_upright" needs only on_shelf.
            up_z = matrix_from_quat(self._obj.data.root_pose_w[:, 3:7])[:, 2, 2]
            upright = up_z > c.upright_cos_tol
            need_up = (c.place_strategy == "extrinsic") and c.require_upright
            place_done = (on_shelf & upright) if need_up else on_shelf
            close = close | (place_dsc & place_done)

            grasp_ok = ((self._phase == int(PickPlacePhase.GRASP)) & (cf > c.grasp_force_n)) |                        (self._phase != int(PickPlacePhase.GRASP))
            place_ok = (place_dsc & place_done) | (~place_dsc)

            can_advance  = close & grasp_ok & place_ok & ~self._broke
            # Envs that actually progress this step (not already at the final phase).
            self._advanced = can_advance & (self._phase < (NUM_PHASES - 1))
            next_ph      = (self._phase + 1).clamp(max=NUM_PHASES - 1)
            self._phase  = torch.where(can_advance, next_ph, self._phase)
            self._phase_ctr += 1

            # Settle at rack after reaching RELEASE: a gentle placement = at RELEASE
            # (which already required a gentle place contact at PLACE_DESCEND) with
            # force staying UNDER budget. Drop the active-contact lower bound — you
            # release the object, so requiring sustained cf>contact_eps wrongly reset
            # the counter. Decrement (not hard-reset) so the oscillating contact
            # force doesn't prevent a settled placement from registering.
            gentle = (cf < self._f_cmd) & (self._phase == int(PickPlacePhase.RELEASE))
            # Stage-B (require_upright): the bottle must also be standing UPRIGHT to count as
            # settled — otherwise the policy could "succeed" by releasing it leaning/toppled.
            if need_up:
                gentle = gentle & upright
            self._settle_ctr = torch.where(
                gentle, self._settle_ctr + 1, (self._settle_ctr - 1).clamp(min=0)
            )
            self._succeeded = self._settle_ctr >= c.settle_steps

            terminated = self._broke | self._succeeded
            truncated  = self.episode_length_buf >= self.max_episode_length - 1

            # Write to self.extras (the dict DirectRLEnv.step actually returns) — the
            # train script reads res[4]["n_succ"]/["n_brk"]. (Was self._extras, a
            # private dict that never propagated, so logged succ/brk were always 0.)
            self.extras["succ_mask"] = self._succeeded.clone()
            self.extras["brk_mask"]  = self._broke.clone()
            self.extras["n_succ"]    = float(self._succeeded.sum().item())
            self.extras["n_brk"]     = float(self._broke.sum().item())
            return terminated, truncated

        # ── Reset ─────────────────────────────────────────────────────────────
        def _reset_idx(self, env_ids) -> None:
            # Lazy-init ee body index (data unavailable during _setup_scene)
            if self._ee_idx < 0:
                self._ee_idx = list(self._robot.data.body_names).index("panda_hand")
                bn = list(self._robot.data.body_names)
                self._lf_idx = bn.index("panda_leftfinger")
                self._rf_idx = bn.index("panda_rightfinger")
            super()._reset_idx(env_ids)
            jp = self._robot.data.default_joint_pos[env_ids].clone()
            # Pre-close the gripper fingers onto the object's half-width so it starts
            # gripped (then the grip force + friction hold it).
            jp[:, 7:9] = 0.022
            jv = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])
            self._robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

            # Place-only: begin already holding the object at transport altitude;
            # the episode does TRANSPORT -> PLACE_DESCEND -> RELEASE only.
            start_phase = int(PickPlacePhase.LIFT) if self.cfg.place_only else 0
            self._phase[env_ids]       = start_phase
            self._phase_ctr[env_ids]   = 0
            self._settle_ctr[env_ids]  = 0
            self._vert_ctr[env_ids]    = 0
            self._best_tilt[env_ids]   = 3.1416
            self._broke[env_ids]       = False
            self._succeeded[env_ids]   = False
            self._set_reset[env_ids]   = True
            self._cf_filt[env_ids]     = 0.0
            self._warmup[env_ids]      = self.cfg.warmup_substeps
            self._az_filt[env_ids]     = -1.0
            # Gripper closed when starting in carry/place mode (holding the object).
            self._gripper_cmd[env_ids] = -1.0 if self.cfg.place_only else 1.0
            self._jt_target[env_ids]   = jp[:, self._arm_ids]

            # Re-place the dynamic object at its init pose (base origin, transport
            # altitude) with zero velocity. DirectRLEnv._reset_idx does NOT reset
            # rigid-object poses for us, so without this the object stays wherever
            # the previous episode (or, in the RTX render, the warmup app.update()s)
            # left it — e.g. fallen onto the shelf — and the warmup seat then has to
            # recover it. Resetting it here makes every reset deterministic and lets
            # the warmup seat grab it cleanly into the gripper.
            obj_state = self._obj.data.default_root_state[env_ids].clone()
            obj_state[:, 0:3] += self.scene.env_origins[env_ids]
            self._obj.write_root_pose_to_sim(obj_state[:, 0:7], env_ids=env_ids)
            self._obj.write_root_velocity_to_sim(obj_state[:, 7:13], env_ids=env_ids)

            self._sample_episode(env_ids)

else:
    # Alias so imports work even without Isaac Sim
    FrankaPickPlaceEnv = MockPickPlaceEnv  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Gym registration
# ─────────────────────────────────────────────────────────────────────────────
gym.register(
    id="FORGE-PickPlace-v0",
    entry_point="forge_plus.isaac_pick_place_env:FrankaPickPlaceEnv",
    kwargs={"cfg": PickPlaceEnvCfg()},
)

gym.register(
    id="FORGE-PickPlace-Mock-v0",
    entry_point="forge_plus.isaac_pick_place_env:MockPickPlaceEnv",
)
