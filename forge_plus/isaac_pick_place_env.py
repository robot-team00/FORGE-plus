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

    # ── Force-signature recovery (proposal §07) ───────────────────────────────
    # A lateral approach error that makes the bottle WEDGE on the cell rim (a jam
    # the recovery loop must clear). 0.0 = no induced jam (normal insertion).
    jam_dx:        float = 0.0     # induced base-aim x error (m) -> rim wedge
    jam_dy:        float = 0.0     # induced base-aim y error (m)
    rec_lift:      float = 0.11    # retract_and_reapproach: how far to lift the EE (m) — clear the rim
    rec_lat:       float = 0.02    # wiggle_search lateral amplitude (m)
    rec_dur_steps: int   = 25      # control steps a recovery maneuver runs before clearing
    jam_force_n:   float = 6.0     # contact >= this with no descent => jam (absolute floor)
    jam_force_frac: float = 0.18   # ...or >= this fraction of F_max (whichever is larger). Low
                                   # enough to catch a wedge WELL below break (robust F_max 72 ->
                                   # ~13 N, above clean seating transients, below the ~17 N wedge).
    jam_progress_mm: float = 2.0   # net base descent (mm) over the window below which it's "stuck"
    jam_window:    int   = 40      # steps of SUSTAINED no-progress to declare a jam. Long enough
                                   # that a clean insertion's brief mid-descent stalls (the policy
                                   # pausing against contact, ~15-25 steps) don't false-trip; a real
                                   # wedge never descends, so it still trips.

    # ── FORGE-style LEARNED insertion (no scripted waypoints, no base-aim) ─────
    # When True: the PPO policy outputs the EE motion (xyz delta) through the OSC
    # controller and must LEARN to align + insert from force + relative-goal obs.
    # The robot is reset to a randomized APPROACH pose above the cell (initial-state
    # setup, exactly as FORGE) — the learned skill is everything after that.
    forge_mode:       bool  = False
    # Curriculum (start EASY so success is discovered, then widen via follow-up runs):
    forge_approach_z: float = 0.68   # EE hand-off height. The lean drops the base ~0.17 below the EE,
                                     # so ee_z 0.68 -> bottle base ~0.51 = right AT the cell entrance
                                     # (top 0.50, floor 0.40). The scripted setup stops here (no descent
                                     # into the cell); the LEARNED policy then does the FULL descent +
                                     # gentle force-controlled insertion down to the floor.
    forge_start_lat:  float = 0.015  # ± random lateral start offset (m) the policy must correct
    forge_setup_steps: int  = 80     # substeps to drive the EE to the entrance pose (short scripted intro)
    forge_act_range:  float = 0.03   # per-step EE delta the policy commands (smaller = gentler, fewer blowups)
    forge_lam:        float = 0.004  # per-step EE motion cap during the LEARNED insertion (m). Small =
                                     # slow approach -> low impact impulse at contact (the 58N spikes that
                                     # broke the glass were impact momentum, not steady force). Setup still
                                     # uses the fast c.lam to traverse to the cell.
    keypoint_k:       float = 60.0   # PBRS reward scale — must dominate the motion penalties so the
                                     # policy descends to seat instead of freezing to dodge penalties
    force_pen_beta:   float = 2.5    # FORGE force-overshoot penalty weight — must be strong enough
                                     # that pushing over budget costs more than the success it buys,
                                     # so the policy learns to seat GENTLY instead of forcing it.
    forge_pos_k:      float = 120.0  # COMPLIANT stiffness during the learned insertion (setup uses
                                     # stiff 400 for reach). Low stiffness keeps contact force
                                     # = k·penetration small -> gentle on the fragile bottle.
    # Compliant contact on the RACK: makes the bottle↔rack (and hand↔rack) contact a
    # SOFT spring (force = k·penetration) instead of a rigid impulse — kills the
    # impulsive force spikes that break the fragile glass. On the rack only, so the
    # bottle↔gripper grip stays rigid (the grasp still holds).
    contact_stiffness: float = 900.0    # compliant_contact_stiffness (N/m); 0 = rigid. Soft enough that
                                        # even peak penetrations keep the bottle↔rack force under the
                                        # ~12 N glass break floor.
    contact_damping:   float = 60.0     # compliant_contact_damping
    # ── Learned safe-release (forge_release_mode) ───────────────────────────────
    # 8th action dim lets the policy command the gripper release; success now requires it
    # to LET GO with the bottle settled (in-cell, at floor, upright, low velocity).
    forge_release_mode: bool  = False
    release_upright_cos: float = 0.90   # cos(tilt) above this (~26 deg) counts as upright on release
                                        # (a bottle standing in a rack cell leans modestly)
    release_vel_tol:     float = 0.25   # bottle speed (m/s) below this counts as settled
    release_hold:        int   = 6      # steps the released+seated state must hold for success
    release_grace:       int   = 8      # steps after release before a not-in-cell bottle = failure
    bad_release_pen:     float = 8.0    # penalty for releasing and losing the bottle (tips/falls out)
    release_floor_tol:   float = 0.10   # |base_z - cell_floor_z| under this counts as resting in the cell
                                        # (the bottle naturally rests a few cm above the nominal floor)
    release_clear_dist:  float = 0.20   # EE must be this far from the bottle BASE (hand RETRACTED
                                        # clear) for success — reached as soon as the moderate retract
                                        # completes (~0.13 is just the grip geometry, so 0.20 = real gap).
    forge_hybrid_retract: bool  = False # after the LEARNED release, drive the hand to a clear pose
                                        # with a fixed retract motion (not a manipulation skill — just
                                        # clearing out so the let-go is visible). Release stays learned.

    forge_no_term:    bool  = False  # render-only: never auto-terminate (so the seated bottle isn't
                                     # reset away before the camera captures the release/retract)
    render_minimal:   bool  = False  # render-only: skip the filtered insert-sensor + compliant-material
                                     # binding (extra SDG-graph state not needed to capture frames)
    forge_obj_cls:    int   = 2      # fix the training object (2=metal_plate, robust) so breakage does
                                     # not derail learning the SEAT; -1 = randomize. Fragility curriculum
                                     # (transfer to glass) is a follow-up once seating is learned.

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
            # forge_release_mode adds an 8th action dim (learned gripper release). Set the
            # gym action_space BEFORE DirectRLEnv reads it.
            if getattr(cfg, "forge_release_mode", False):
                cfg.action_space = 8
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
            self._lf_idx:   int = -1  # left finger body (held-object grasp centre)
            self._rf_idx:   int = -1  # right finger body
            self._grip_ids  = [7, 8]  # actuated gripper joint ids (resolved lazily per gripper)
            self._osc_init: bool = False

            # Actions / EE state. In forge_release_mode the policy gets an 8th action dim
            # (action[7] = learned gripper release command); otherwise 7 (arm-only).
            _adim = 8 if self.cfg.forge_release_mode else 7
            self._actions     = torch.zeros(N, _adim, device=d)
            self._prev_actions = torch.zeros(N, _adim, device=d)  # for action-rate smoothness
            self._ee_quat_des = torch.zeros(N, 4, device=d)
            self._ee_quat_des[:, 0] = 1.0
            self._gripper_cmd = torch.ones(N, device=d)   # +1 = open, -1 = closed
            # forge_release: latched "policy commanded release" + age since release (grace).
            self._released   = torch.zeros(N, dtype=torch.bool, device=d)
            self._newly_released = torch.zeros(N, dtype=torch.bool, device=d)  # released THIS step
            self._rel_age    = torch.zeros(N, dtype=torch.long, device=d)

            # Contact force (low-pass filter)
            self._cf_filt  = torch.zeros(N, device=d)
            self._cf_insert = torch.zeros(N, device=d)   # filtered bottle↔rack insertion force only
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
            self._bad_release = torch.zeros(N, dtype=torch.bool, device=d)  # released but lost the bottle
            self._drop_quality = torch.zeros(N, device=d)   # dense release-quality shaping signal
            self._rel_upz     = torch.zeros(N, device=d)    # uprightness diagnostic
            self._retract_prog = torch.zeros(N, device=d)   # dense hand-retract progress (post-release)
            self._prev_eod    = torch.zeros(N, device=d)    # previous EE->object distance
            self._succeeded  = torch.zeros(N, dtype=torch.bool, device=d)
            self._advanced   = torch.zeros(N, dtype=torch.bool, device=d)  # advanced a phase this step
            self._set_reset  = torch.zeros(N, dtype=torch.bool, device=d)
            self._warmup     = torch.zeros(N, dtype=torch.long, device=d)
            # Distance from the hand origin to the grasp point (between the fingertip
            # pads) along the hand's local +z. Per-env so it can be swept/calibrated;
            # 0.067 seats the block centrally between the panda pads (calibrated).
            # The Robotiq 2F-140's pads sit much lower below the flange (calibrated
            # empirically via the headless recovery probe).
            _tcp = 0.19 if self.cfg.gripper == "robotiq_2f140" else 0.067
            self._grasp_tcp_d = torch.full((N,), _tcp, device=d)
            # Gripper-length delta vs the franka hand: shifts the FORGE hand-off /
            # transport / retract HEIGHTS so the BOTTLE traverses the same altitudes.
            self._tcp_dz = _tcp - 0.067
            # Finger half-opening to rest the pads at during warmup: block half-width
            # (0.020) minus 1 mm so the pads sit just at the surface (clean seat, no
            # deep penetration), then the PD grip takes over and holds by friction.
            self._grasp_seat_w = 0.006
            # Robotiq finger_joint targets (rad; 0 = CLOSED, 0.785 = fully open —
            # inverted vs the ROS URDF). Calibration curve (probe_robotiq_env):
            # pad-body separation 0.040 m at 0 -> 0.127 m at 0.7; the ~1.6 cm bottle
            # neck maps to ~0.09; squeeze a little under, seat a little over.
            self._rq_close = 0.05
            self._rq_seat  = 0.10
            self._rq_open  = 0.45
            # PD gains for the position-controlled grip (effort = k·(target-pos) - kd·vel).
            # k·overlap sets the squeeze: 1500 N/m · 0.010 m = 15 N grip (friction
            # 1.6·15 = 24 N >> the 0.5 N block weight), penetration sub-mm under rigid contact.
            self._grip_pos_ks = 1500.0
            self._grip_pos_kd = 40.0
            self._az_filt    = torch.full((N,), -1.0, device=d)
            self._jt_target  = torch.zeros(N, 7, device=d)
            self._extras: dict = {}

            # ── Recovery state (force-signature LLM recovery; see RecoveryLoop) ──
            self._rec_off    = torch.zeros(N, 3, device=d)   # world-frame OSC target offset
            self._rec_steps  = torch.zeros(N, dtype=torch.long, device=d)  # maneuver countdown
            self._rec_wiggle = torch.zeros(N, dtype=torch.bool, device=d)  # lateral search active
            self._rec_open   = torch.zeros(N, dtype=torch.bool, device=d)  # force fingers open (regrasp)
            self._jam_on     = torch.zeros(N, dtype=torch.bool, device=d)  # induced misalignment active
            self._last_rec_action = None     # last recovery primitive applied (HUD)

            # ── FORGE-mode state (learned insertion) ──
            self._skill_policy = None     # optional loaded policy that drives step_skill (forge)
            self._setup_ctr  = torch.zeros(N, dtype=torch.long, device=d)  # approach-pose setup countdown
            self._start_off  = torch.zeros(N, 2, device=d)                 # random lateral start offset
            self._best_dist  = torch.full((N,), 9.9, device=d)             # best (min) dist-to-goal this episode
            self._prev_dist  = torch.full((N,), 9.9, device=d)             # PBRS: previous-step dist-to-goal
            self._min_dist   = torch.full((N,), 9.9, device=d)             # best (min) dist achieved this episode
            self._rec_phase  = torch.zeros(N, dtype=torch.long, device=d)  # wiggle phase counter
            self._jam_cooldown = torch.zeros(N, dtype=torch.long, device=d)  # suppress jam-detect after a recovery
            # Contact + base-height history for the force signature (ring buffers).
            self._sig_len    = 64
            self._cf_hist    = torch.zeros(N, self._sig_len, device=d)
            self._basez_hist = torch.zeros(N, self._sig_len, device=d)
            self._latx_hist  = torch.zeros(N, self._sig_len, device=d)
            self._laty_hist  = torch.zeros(N, self._sig_len, device=d)
            self._sig_ptr    = 0

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

            # PARSE GHOST (robotiq only, spawned BEFORE the robot — parse order matters,
            # matching the healthy probes): a standalone 2F-140 articulation parked far
            # below the workspace, gravity-free, never driven or rendered in-frame.
            # Its presence makes PhysX materialize the merged gripper's loop joints.
            if self.cfg.gripper == "robotiq_2f140":
                from isaaclab.actuators import ImplicitActuatorCfg
                self._ghost = Articulation(ArticulationCfg(
                    prim_path="/World/GhostGripper",   # OUTSIDE the cloned env namespace (matches the healthy probes)
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=("/workspace/assets/isaac51/Robots/Robotiq/2F-140/"
                                  "Robotiq_2F_140_physics_edit.usd"),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.0, -5.0),
                        joint_pos={"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
                                   ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0}),
                    actuators={"all_passive": ImplicitActuatorCfg(
                        joint_names_expr=[".*"], effort_limit_sim=1.0,
                        velocity_limit_sim=2.0, stiffness=0.0, damping=0.01,
                        friction=0.0, armature=0.0)},
                ))

            # Robot
            robot_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
            robot_cfg.spawn.activate_contact_sensors = True
            if self.cfg.gripper == "robotiq_2f140":
                # Franka + Robotiq 2F-140 (scripts/build_franka_robotiq_2f140.py). Two
                # PhysX constraints shape everything here — see NOTE 2/3 in that script:
                #  * TELEPORT CONTRACT: the gripper's four-bar loop joints do not survive
                #    write_joint_state_to_sim. Spawn at the authored DEFAULT arm pose (no
                #    env-pose overrides) and drive by targets only; the FORGE setup (OSC)
                #    takes the arm from the default pose to the hand-off.
                #  * PARSE GHOST: the merged gripper's loop joints only materialize when a
                #    standalone 2F-140 articulation is also in the scene (spawned below).
                robot_cfg.spawn.usd_path = ("/workspace/assets/isaac51/Robots/"
                                            "FrankaRobotics/FrankaPanda/franka_robotiq_2f140.usd")
                _jp = {k: v for k, v in robot_cfg.init_state.joint_pos.items()
                       if not k.startswith("panda_finger")}
                _jp.update({"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
                            ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0})
                robot_cfg.init_state.joint_pos = _jp
                # Actuators: isaaclab's UR10e+2F-140 template. Drive the finger_joint;
                # the pad springs adapt; the rest is loop-/mimic-owned (passive).
                from isaaclab.actuators import ImplicitActuatorCfg
                robot_cfg.actuators.pop("panda_hand", None)
                robot_cfg.actuators["gripper_drive"] = ImplicitActuatorCfg(
                    joint_names_expr=["finger_joint"], effort_limit_sim=10.0,
                    velocity_limit_sim=1.0, stiffness=11.25, damping=0.1,
                    friction=0.0, armature=0.0)
                robot_cfg.actuators["gripper_finger"] = ImplicitActuatorCfg(
                    joint_names_expr=[".*_inner_finger_joint"], effort_limit_sim=1.0,
                    velocity_limit_sim=1.0, stiffness=0.2, damping=0.001,
                    friction=0.0, armature=0.0)
                robot_cfg.actuators["gripper_passive"] = ImplicitActuatorCfg(
                    joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint",
                                      "right_outer_knuckle_joint"],
                    effort_limit_sim=1.0, velocity_limit_sim=2.0, stiffness=0.0,
                    damping=0.0, friction=0.0, armature=0.0)
            else:
                robot_cfg.spawn.usd_path = "/workspace/assets/franka/panda_instanceable.usd"
                _jp = dict(robot_cfg.init_state.joint_pos)
                _jp["panda_joint2"] = -0.73
                _jp["panda_joint4"] = -2.46
                _jp["panda_joint6"] = 2.85
                robot_cfg.init_state.joint_pos = _jp
            # Disable the joint PD controllers on the proximal joints so the OSC
            # (Jacobian-transpose effort targets) actually moves the arm instead of
            # being overpowered back to the default pose. Mirrors FrankaPlaceEnv.
            # Effort control for the OSC: zero the joint-position stiffness so the
            # actuator PD doesn't fight the OSC, but KEEP joint-velocity damping --
            # without it the torque-controlled joints buzz at high frequency (the
            # vibration). The damping opposes joint velocity and kills the buzz.
            # ROBOTIQ EXCEPTION: spawn with the DEFAULT holding gains — a limp arm
            # free-falls during initialization and the sag TEARS the gripper's four-bar
            # loop joints (verified: probe_rq_scene ZEROARM=1). The gains are zeroed at
            # runtime in _reset_idx via a joint-PARAMETER write (moves no bodies).
            if self.cfg.gripper != "robotiq_2f140":
                for _an in ("panda_shoulder", "panda_forearm"):
                    robot_cfg.actuators[_an].stiffness = 0.0
                    robot_cfg.actuators[_an].damping = 80.0
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
                    ),  # compliant contact material applied post-spawn (UsdFileCfg rejects it)
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(self.cfg.rack_x, self.cfg.rack_y, self.cfg.rack_z),
                    ),
                )
            )

            # Contact sensor on hand + fingers, filtered to the object and rack so
            # net_forces_w reports the grasp/place contact force (panda_hand alone
            # never touches either surface → was reading ~0 N). For the Robotiq the
            # touching bodies are the finger pads/links (regex segments cannot span
            # '/', and the bare hand reads ~0 N, so sense the four finger bodies).
            _sensor_expr = ("/World/envs/env_.*/Robot/Robotiq_2F_140_edit/(left|right)_(inner|outer)_finger"
                            if self.cfg.gripper == "robotiq_2f140" else
                            "/World/envs/env_.*/Robot/panda_(hand|leftfinger|rightfinger)")
            import os as _os
            _bisect = int(_os.environ.get("RQ_BISECT", "0"))   # TEMP four-bar bisect
            self._contact_sensor = None if _bisect >= 3 else ContactSensor(
                ContactSensorCfg(
                    prim_path=_sensor_expr,
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
            self._surf_sensor = None if _bisect >= 2 else ContactSensor(
                ContactSensorCfg(
                    prim_path="/World/envs/env_.*/(Object|Rack)",
                    update_period=0.0,
                    history_length=1,
                    track_air_time=False,
                )
            )

            # INSERTION-ONLY sensor: the pairwise contact force between the Object and
            # the Rack (force_matrix_w). This is the TRUE insertion force that should
            # gate breakage — it excludes the grip (gripper↔object), hand↔rack bumps,
            # and the impulse artifacts that pollute the whole-body surf sensor.
            # (FORGE gates on the arm EE force J†·τ_ext for the same reason.)
            # Sensor on the RACK (which has the contact-reporter API; the bottle USD
            # does not), filtered to the Object -> force on the rack from the bottle =
            # the insertion contact (Newton's 3rd law, same magnitude).
            if self.cfg.render_minimal or _bisect >= 2:
                self._insert_sensor = None   # render: skip the extra filtered sensor
            else:
                self._insert_sensor = ContactSensor(
                    ContactSensorCfg(
                        prim_path="/World/envs/env_.*/Rack",
                        update_period=0.0,
                        history_length=1,
                        track_air_time=False,
                        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                    )
                )

            # Register with scene
            self.scene.articulations["robot"]  = self._robot
            if getattr(self, "_ghost", None) is not None:
                # register the parse ghost so it initializes through the same
                # InteractiveScene pipeline as the robot (view-creation ordering)
                self.scene.articulations["ghost"] = self._ghost
            self.scene.rigid_objects["table"]  = self._table
            self.scene.rigid_objects["object"] = self._obj
            self.scene.rigid_objects["rack"]   = self._rack
            if self._contact_sensor is not None:
                self.scene.sensors["contact"]  = self._contact_sensor
            if self._surf_sensor is not None:
                self.scene.sensors["surf"]     = self._surf_sensor
            if self._insert_sensor is not None:
                self.scene.sensors["insert"]   = self._insert_sensor
            # ── Compliant contact on the rack (soft spring, not rigid impulse) ──
            if self.cfg.contact_stiffness > 0 and not self.cfg.render_minimal and _bisect < 1:
                self._apply_rack_compliance()
            self.scene.clone_environments(copy_from_source=False)

            # Cache joint / body indices
            self._arm_ids = list(range(7))
            self._ee_idx  = -1  # resolved lazily in _reset_idx (data not ready at setup time)

        def _apply_rack_compliance(self) -> None:
            """Bind a COMPLIANT-contact physics material to the rack collisions, so the
            bottle↔rack contact is a soft spring (force = k·penetration) instead of a
            rigid impulse. Applied to the source env_0 rack before cloning."""
            try:
                import omni.usd
                from pxr import UsdShade, UsdPhysics, PhysxSchema
                stage = omni.usd.get_context().get_stage()
                matp = "/World/CompliantRackMat"
                mat = UsdShade.Material.Define(stage, matp)
                UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                fr = UsdPhysics.MaterialAPI(mat.GetPrim())
                fr.CreateStaticFrictionAttr(0.8); fr.CreateDynamicFrictionAttr(0.8)
                px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
                px.CreateCompliantContactStiffnessAttr(float(self.cfg.contact_stiffness))
                px.CreateCompliantContactDampingAttr(float(self.cfg.contact_damping))
                n = 0
                for prim in stage.Traverse():
                    pth = prim.GetPath().pathString
                    if "/Rack" in pth and prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                            mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                            materialPurpose="physics")
                        n += 1
                print(f"[compliant] bound soft-contact material (k={self.cfg.contact_stiffness}) "
                      f"to {n} rack collision prims", flush=True)
            except Exception as exc:
                print(f"[compliant] FAILED to apply rack compliance: {exc}", flush=True)

        # ── Episode sampling ──────────────────────────────────────────────────
        def _sample_episode(self, ids) -> None:
            n   = len(ids)
            cls = torch.randint(0, self._n_obj_cls, (n,), device=self.device)
            if self.cfg.forge_mode and self.cfg.forge_obj_cls >= 0:
                cls = torch.full((n,), self.cfg.forge_obj_cls, device=self.device, dtype=torch.long)
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

        def _raw_insertion_force(self) -> torch.Tensor:
            """Pairwise Object↔Rack contact force magnitude (the TRUE insertion force).
            Excludes the grip, hand↔rack bumps, and other-body impulses."""
            if getattr(self, "_insert_sensor", None) is None:
                return torch.zeros(self.num_envs, device=self.device)
            data = getattr(self._insert_sensor, "data", None)
            fm = getattr(data, "force_matrix_w", None) if data is not None else None
            if fm is not None and fm.numel() > 0:
                return torch.norm(fm.reshape(self.num_envs, -1, 3), dim=-1).sum(dim=1)
            return torch.zeros(self.num_envs, device=self.device)

        def _contact_force(self) -> torch.Tensor:
            """Low-pass filtered scalar contact force (whole object+rack surf sensor)."""
            return self._cf_filt

        def _insertion_force(self) -> torch.Tensor:
            """Low-pass filtered Object↔Rack insertion force (gates breakage in forge)."""
            return self._cf_insert

        def _ee_arm_force(self) -> torch.Tensor:
            """Force the ARM feels at the wrist (the reaction wrench at the EE joint),
            FORGE-style — immune to the bottle↔rack contact-solver artifact because it
            reflects what the robot actually transmits, not the solver's penetration
            impulse. Returns the linear force magnitude; 0 if the API is unavailable."""
            d = getattr(self._robot, "data", None)
            w = getattr(d, "body_incoming_joint_wrench_b", None)
            if w is not None and self._ee_idx >= 0 and w.numel() > 0:
                return torch.norm(w[:, self._ee_idx, :3], dim=-1)
            return torch.zeros(self.num_envs, device=self.device)

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

        def _raw_contact_vec3(self) -> torch.Tensor:
            """Net contact force VECTOR (N,3) applied to the object/rack (world)."""
            sdata = getattr(self._surf_sensor, "data", None)
            snf = getattr(sdata, "net_forces_w", None) if sdata is not None else None
            if snf is not None and snf.numel() > 0:
                return snf.sum(dim=1)
            return torch.zeros(self.num_envs, 3, device=self.device)

        def f_cmd_norm(self) -> torch.Tensor:
            """Normalised F_cmd in [0, 1] for policy conditioning."""
            return (self._f_cmd / 120.0).unsqueeze(-1)

        # ══ FORGE-style learned insertion (no scripted waypoints / base-aim) ══════
        def _forge_goal_w(self) -> torch.Tensor:
            """Seated-pose target (cell center at the floor) in world frame."""
            g = self.scene.env_origins.clone()
            g[:, 0] += self.cfg.rack_x
            g[:, 1] += self.cfg.rack_y
            g[:, 2] += self.cfg.cell_floor_z
            return g

        def _forge_targets(self, ee_pos_w: torch.Tensor):
            """EE target for FORGE mode. During the (non-learned) setup window the EE
            is driven to a randomized APPROACH pose above the cell; after that the
            LEARNED policy's xyz delta drives the alignment + insertion."""
            c = self.cfg
            orig = self.scene.env_origins
            in_setup = (self._setup_ctr > 0).unsqueeze(-1)
            # SETUP (positioning only — NOT the learned skill): drive the GRIPPER over
            # the cell via the reachable UP-OVER-DOWN path (a direct diagonal drive
            # saturates the arm's workspace ~12 cm short in y). Stage 1: up & over to
            # the cell xy at transport height. Stage 2: straight down to the approach
            # height. NO base-aim — so the bottle's base starts ~6 cm off (the lean),
            # and the LEARNED policy must correct that alignment itself + descend + force.
            stage1 = self._setup_ctr > (c.forge_setup_steps // 2)
            appr = ee_pos_w.clone()
            appr[:, 0] = orig[:, 0] + c.rack_x + self._start_off[:, 0]
            appr[:, 1] = orig[:, 1] + c.rack_y + self._start_off[:, 1]
            # _tcp_dz shifts the HAND heights for longer grippers (robotiq) so the
            # BOTTLE traverses/hands-off at the same altitudes as with the panda hand.
            appr[:, 2] = orig[:, 2] + torch.where(stage1, c.transport_z, c.forge_approach_z) + self._tcp_dz
            pol = ee_pos_w + self._actions[:, :3] * c.forge_act_range  # learned EE delta (gentle)
            raw = torch.where(in_setup, appr, pol)
            # Fast traverse during setup, SLOW (low-impulse) approach during the learned
            # insertion so the bottle never hits the cell with breaking momentum.
            lam = torch.where(in_setup, torch.full_like(appr[:, :1], c.lam),
                              torch.full_like(appr[:, :1], c.forge_lam))
            # After the policy releases, the bottle is placed — the hand no longer needs the
            # slow gentle-insertion motion cap, so retract at a MODERATE speed (the full setup
            # lam yanks the arm and spikes joint forces; 1.5 cm/step pulls clear smoothly).
            if self.cfg.forge_release_mode:
                lam = torch.where(self._released.unsqueeze(-1),
                                  torch.full_like(lam, 0.015), lam)
            delta = torch.maximum(torch.minimum(raw - ee_pos_w, lam), -lam)
            target = ee_pos_w + delta
            # ── FORGE force authority (safety, NOT the policy): when axial contact
            # reaches the per-object budget, FREEZE the descent (don't push the EE
            # lower) — but leave xy free so the policy can still slide/align, and
            # don't retreat (so resting on the cell floor = seated, not undone).
            # Compliant stiffness (forge_pos_k) keeps lateral rams survivable. This
            # bounds the force to ~F_max without preventing the gentle insertion.
            budget = self._obj_budget[self._obj_cls]
            over = (self._cf_insert > budget) & (self._setup_ctr == 0)   # insertion force only
            if self.cfg.forge_release_mode:
                over = over & (~self._released)   # after release, never freeze z — let the hand lift away
            frozen_z = torch.maximum(target[:, 2], ee_pos_w[:, 2])
            target[:, 2] = torch.where(over, frozen_z, target[:, 2])

            # ══ Force-signature recovery hooks on the LEARNED insertion ═══════════
            # Active only in the recovery demo: either a jam is configured OR a recovery maneuver
            # is currently running. Zero overhead / no effect on the normal trained policy, which
            # runs with jam_dx = jam_dy = 0 and never triggers a recovery (_rec_steps stays 0).
            rec_active = self._rec_steps > 0
            if c.jam_dx != 0.0 or c.jam_dy != 0.0 or bool(rec_active.any()):
                orig   = self.scene.env_origins
                base_w = self._obj.data.root_pose_w[:, :3]
                # (a) RECOVERY maneuver: while a recovery is running, drive the hand back to the
                # TRAINING HAND-OFF pose (the cell-entrance approach the setup delivers to, incl.
                # the same start offset) + the recovery's lateral offset (wiggle / rotate_align).
                # Two rules learned the hard way:
                #   * the pose must be the hand-off (incl. _start_off), NOT hand-off + rec_lift —
                #     an extra lift puts the policy OUT of its training distribution and it never
                #     re-descends (it only knows descents from the entrance);
                #   * the drive rate must be the setup's c.lam — the proven bottle-carrying rate.
                #     A gentler clamp (0.012 ~ 4.8 N of impedance pull at kpos 400) is eaten by
                #     the unmodeled bottle payload and the lift never rises.
                # When the maneuver expires the LEARNED policy re-descends exactly as it does
                # after setup — now aligned, because the jam was cleared.
                if bool(rec_active.any()):
                    rec = self._rec_off.clone()
                    if bool(self._rec_wiggle.any()):
                        wig = torch.zeros_like(rec)
                        wig[:, 0] = torch.sin(self._rec_phase.float() * 0.5) * c.rec_lat
                        rec = torch.where(self._rec_wiggle.unsqueeze(-1), rec + wig, rec)
                    clear = ee_pos_w.clone()
                    clear[:, 0] = orig[:, 0] + c.rack_x + self._start_off[:, 0] + rec[:, 0]
                    clear[:, 1] = orig[:, 1] + c.rack_y + self._start_off[:, 1] + rec[:, 1]
                    clear[:, 2] = orig[:, 2] + c.forge_approach_z + self._tcp_dz
                    step = (clear - ee_pos_w).clamp(-c.lam, c.lam)
                    target = torch.where(rec_active.unsqueeze(-1), ee_pos_w + step, target)
                # (b) INDUCED JAM (only when not recovering): base-aim the bottle at an OFF-CENTER
                # wedge point so it wedges on the cell rim (high contact, no descent -> is_failure).
                jam_active = self._jam_on & (self._setup_ctr == 0) & (~rec_active)
                if bool(jam_active.any()):
                    wedge_x = orig[:, 0] + c.rack_x + c.jam_dx
                    wedge_y = orig[:, 1] + c.rack_y + c.jam_dy
                    aim_x = wedge_x - (base_w[:, 0] - ee_pos_w[:, 0])   # EE so the BASE lands off-center
                    aim_y = wedge_y - (base_w[:, 1] - ee_pos_w[:, 1])
                    tx = ee_pos_w[:, 0] + (aim_x - ee_pos_w[:, 0]).clamp(-c.forge_lam, c.forge_lam)
                    ty = ee_pos_w[:, 1] + (aim_y - ee_pos_w[:, 1]).clamp(-c.forge_lam, c.forge_lam)
                    m = jam_active
                    target[m, 0] = tx[m]
                    target[m, 1] = ty[m]

            # Wrist: STIFF during setup (hold upright while traversing), COMPLIANT during
            # the learned insertion. Best config found: a firm wrist levers a large force
            # against the constrained bottle (the soft contact only lets it penetrate
            # further, not lower the force), so compliant wins — successful insertions
            # then keep contact ~5N. (Residual: ~90% of attempts still spike >F_break
            # somewhere and the policy drifts; not fully solved.)
            # Wrist orientation stiffness: firm (400) during the scripted setup, MODERATE (110)
            # during the learned descent so the bottle stays UPRIGHT as it's pushed into the cell
            # (the old 40 let it lean ~40 deg over the longer descent from the entrance hand-off).
            ori_k = torch.where(self._setup_ctr > 0,
                                torch.full((self.num_envs,), 400.0, device=self.device),
                                torch.full((self.num_envs,), 110.0, device=self.device))
            # Once the policy has LEARNED-released, HOLD the arm still at its release pose. The
            # policy (trained to insert) otherwise keeps driving the EE down and pushes on the
            # just-freed bottle (~47 N); freezing the arm lets the bottle settle cleanly with the
            # open gripper clear. The arm simply lets go and stops — no pull-away.
            if self.cfg.forge_release_mode and not self.cfg.forge_hybrid_retract:
                rel = self._released
                target = torch.where(rel.unsqueeze(-1), ee_pos_w, target)
                ori_k = torch.where(rel,
                                    torch.full((self.num_envs,), 200.0, device=self.device), ori_k)
            # HYBRID retract: after the LEARNED release, pull the (now open, empty) hand to a
            # MODERATE clear pose — up and slightly back so it's clearly off the bottle, but NOT
            # all the way up to a ceiling/home pose — then hold there. Post-task clearing, not a
            # manipulation skill; the bottle stays seated (the gripper genuinely opened).
            if self.cfg.forge_release_mode and self.cfg.forge_hybrid_retract:
                rel = self._released
                age = self._rel_age
                # Phase A (settle): hold briefly so the freed bottle separates from the open pads.
                settling   = (rel & (age <= 5)).unsqueeze(-1)
                # Phase B (retract): drive to a moderate clear pose and stop (the clamp eases in).
                retracting = (rel & (age > 5)).unsqueeze(-1)
                clear = ee_pos_w.clone()
                clear[:, 0] = orig[:, 0] + 0.30                 # step back ~15 cm from the cell
                clear[:, 1] = orig[:, 1] + c.rack_y
                clear[:, 2] = orig[:, 2] + 0.62 + self._tcp_dz  # lift to just above the bottle (~22 cm)
                rstep = ee_pos_w + (clear - ee_pos_w).clamp(-0.035, 0.035)  # brisk pull-away
                target = torch.where(settling, ee_pos_w, target)
                target = torch.where(retracting, rstep, target)
                ori_k = torch.where(rel,
                                    torch.full((self.num_envs,), 200.0, device=self.device), ori_k)
            self._setup_ctr = (self._setup_ctr - 1).clamp(min=0)
            return target, ori_k

        def _forge_get_observations(self) -> dict:
            r = self._robot
            jp = r.data.joint_pos[:, self._arm_ids]
            jv = r.data.joint_vel[:, self._arm_ids]
            ee_p = r.data.body_pos_w[:, self._ee_idx] - self.scene.env_origins
            ee_q = r.data.body_quat_w[:, self._ee_idx]
            ft = self._contact_wrench_6d()
            base_w = self._obj.data.root_pose_w[:, :3]
            base_to_goal = self._forge_goal_w() - base_w                 # (N,3) relative goal
            obj_up = torch.bmm(matrix_from_quat(self._obj.data.root_pose_w[:, 3:7]),
                               torch.tensor([0., 0., 1.], device=self.device).view(1, 3, 1)
                               .expand(self.num_envs, 3, 1)).squeeze(-1)
            fcmd = self.f_cmd_norm()
            # 7+7+3+4+6+3+3+1 = 34
            return {"policy": torch.cat([jp, jv, ee_p, ee_q, ft, base_to_goal, obj_up, fcmd], dim=-1)}

        def _forge_get_rewards(self) -> torch.Tensor:
            c = self.cfg
            cf = self._insertion_force()       # penalise the TRUE insertion force (matches break gate)
            base = self._obj.data.root_pose_w[:, :3] - self.scene.env_origins
            goal = self._forge_goal_w() - self.scene.env_origins
            d = base - goal
            dist = d.norm(dim=-1)
            live = self._setup_ctr == 0
            # PURE PROGRESS shaping: F = prev_dist − dist (NO discount factor).
            # Telescopes to (dist_0 − dist_final): only NET progress toward the seat
            # pays; staying still earns exactly 0 and oscillating nets ~0, so it
            # cannot be farmed by hovering. (A 0.99 discount here created a positive
            # living reward ∝ dist that the policy farmed by hovering far from goal.)
            valid = self._prev_dist < 9.0
            shape = torch.where(valid, self._prev_dist - dist, torch.zeros_like(dist))
            self._prev_dist = dist.clone()
            r = c.keypoint_k * shape * live.float()
            # force-overshoot penalty (FORGE): penalise contact above the budget.
            excess = ((cf - self._f_cmd).clamp(min=0.0) / self._f_cmd.clamp(min=1.0)).clamp(max=3.0)
            r = r - c.force_pen_beta * excess
            # smoothness + time — kept SMALL so they don't suppress the descent the
            # policy must explore to discover the seat (heavy penalties -> it freezes).
            jvel = self._robot.data.joint_vel[:, self._arm_ids].abs().mean(dim=-1)
            r = r - 0.01 * jvel
            arate = (self._actions - self._prev_actions).abs().mean(dim=-1)
            r = r - 0.03 * arate
            self._prev_actions = self._actions.clone()
            r = r - 0.005
            # terminal cliffs
            r = r + self._succeeded.float() * 50.0
            r = r - self._broke.float() * 6.0
            # forge_release: shape a CLEAN drop + a hands-off retract.
            if c.forge_release_mode:
                rel = self._released.float()
                # (1) SMALL upright-drop reward. Kept small on purpose: a large per-step reward for
                #     merely keeping the bottle placed created a lazy optimum (the policy sat next to
                #     the bottle collecting reward and never retracted). The retract + success terms
                #     below must dominate so the policy is pulled toward letting go and backing away.
                r = r + 0.2 * self._drop_quality
                # (2) penalise the bottle LEANING after release (teaches a vertical placement)
                r = r - 1.5 * rel * (1.0 - self._rel_upz).clamp(min=0.0)
                # (3) penalise the open hand still PUSHING the bottle after release
                r = r - 0.25 * rel * (cf - 3.0).clamp(min=0.0)
                # (4) REWARD retracting the hand away from the bottle after release — the dominant
                #     post-release signal, so a visible pull-away is the only way to earn reward.
                r = r + 20.0 * self._retract_prog
                # (5) losing the bottle entirely (tips out / falls)
                r = r - self._bad_release.float() * c.bad_release_pen
                # (6) RELEASE-LOW: penalise letting go while the bottle is still high above the
                #     cell floor. Without this the policy "cheats" — releases at the hand-off and
                #     lets the bottle drop, instead of DESCENDING + inserting it. This forces the
                #     learned policy to do the actual gentle insertion before it lets go.
                rel_height = (base[:, 2] - c.cell_floor_z).clamp(min=0.0)   # how high at release
                r = r - 12.0 * self._newly_released.float() * rel_height
                # (7) REWARD committing to release once the bottle is descended into the cell —
                #     without this the policy learns to descend and HOLD, only letting go via
                #     exploration noise (so the deterministic/eval policy never releases).
                low = (base[:, 2] - c.cell_floor_z) < 0.06   # must descend to within 6 cm of the floor
                r = r + 10.0 * self._newly_released.float() * low.float()
            # no learning signal during the (non-skill) setup window
            r = torch.where(self._setup_ctr > 0, torch.zeros_like(r), r)
            return r

        def _forge_get_dones(self):
            c = self.cfg
            cf = self._insertion_force()       # gate breakage on the TRUE bottle↔rack insertion force
            base = self._obj.data.root_pose_w[:, :3] - self.scene.env_origins
            in_cell = ((base[:, 0] - c.rack_x).abs() < 0.05) & ((base[:, 1] - c.rack_y).abs() < 0.05)
            live = (self._setup_ctr == 0) & (self._warmup == 0)
            self._broke = (cf > self._f_break) & live
            self._bad_release = torch.zeros_like(self._broke)
            if c.forge_release_mode:
                # SAFE DROP: success requires the policy to LET GO (released) with the bottle
                # resting on the cell floor, upright, and settled (low velocity).
                up_z   = matrix_from_quat(self._obj.data.root_pose_w[:, 3:7])[:, 2, 2].clamp(-1.0, 1.0)
                at_floor = (base[:, 2] - c.cell_floor_z).abs() < c.release_floor_tol
                upright  = up_z > c.release_upright_cos
                bvel     = self._obj.data.root_vel_w[:, :3].norm(dim=-1)
                settled  = bvel < c.release_vel_tol
                # hand must RETRACT clear of the bottle — opening the fingers is not "letting go"
                eo_dist  = (self._robot.data.body_pos_w[:, self._ee_idx]
                            - self._obj.data.root_pose_w[:, :3]).norm(dim=-1)
                hand_clear = eo_dist > c.release_clear_dist
                # dense retract progress (reward moving the hand away after release)
                self._retract_prog = self._released.float() * (eo_dist - self._prev_eod).clamp(-0.1, 0.1)
                self._prev_eod = eo_dist
                seated   = in_cell & at_floor & upright & settled & self._released
                # Only require the hand to be RETRACTED clear when the hybrid retract is enabled.
                # Without it, success = the policy let go and the bottle is left standing.
                if c.forge_hybrid_retract:
                    seated = seated & hand_clear
                # age since release (grace before judging a lost bottle a failure)
                self._rel_age = torch.where(self._released, self._rel_age + 1, self._rel_age)
                # released but the bottle left the cell / fell / tipped after the grace = failure
                lost = self._released & live & (self._rel_age > c.release_grace) & (
                    (~in_cell) | (base[:, 2] < 0.25) | (up_z < 0.7))
                self._bad_release = lost
                # DENSE drop-quality shaping signal (reused in _forge_get_rewards): once let go
                # in the cell near the floor, reward uprightness so the policy gets a gradient
                # toward a clean vertical drop even before the strict success cliff fires.
                good_drop = self._released & in_cell & at_floor & (~lost)
                self._drop_quality = good_drop.float() * up_z.clamp(min=0.0)
                self._rel_upz = up_z   # for diagnostics
            else:
                seated = in_cell & ((base[:, 2] - c.cell_floor_z).abs() < c.insert_depth_tol)
            self._settle_ctr = torch.where(seated & live, self._settle_ctr + 1,
                                           (self._settle_ctr - 1).clamp(min=0))
            self._succeeded = self._settle_ctr >= (c.release_hold if c.forge_release_mode else 6)
            # Dropped the bottle (flung from the grip / fell): terminate so the
            # blown-up state is reset instead of polluting training with huge dist.
            dropped = base[:, 2] < 0.25
            terminated = (self._broke | self._succeeded | self._bad_release | (dropped & live)) & live
            if c.forge_no_term:
                terminated = torch.zeros_like(terminated)   # render: keep the seated bottle in place
            truncated = self.episode_length_buf >= self.max_episode_length - 1
            self.extras["succ_mask"] = self._succeeded.clone()
            self.extras["brk_mask"] = self._broke.clone()
            self.extras["n_succ"] = float(self._succeeded.sum().item())
            self.extras["n_brk"] = float(self._broke.sum().item())
            if c.forge_release_mode:
                self.extras["n_rel"]    = float(self._released.sum().item())     # how many have let go
                self.extras["n_badrel"] = float(self._bad_release.sum().item())  # let go and lost it
                _rl = self._released & live
                self.extras["relupz"] = float((self._rel_upz * _rl.float()).sum().item()
                                              / _rl.float().sum().clamp(min=1.0).item())  # mean uprightness of released
                self.extras["releod"] = float((self._prev_eod * _rl.float()).sum().item()
                                              / _rl.float().sum().clamp(min=1.0).item())  # mean EE->bottle dist after release
            # diagnostics (live envs): how close is the base to the seated pose?
            lm = live.float()
            denom = lm.sum().clamp(min=1.0)
            goal = self._forge_goal_w() - self.scene.env_origins
            dist = (base - goal).norm(dim=-1)
            self._min_dist = torch.where(live, torch.minimum(self._min_dist, dist), self._min_dist)
            self.extras["fdist"] = float((dist * lm).sum().item() / denom.item())
            self.extras["fbz"] = float((base[:, 2] * lm).sum().item() / denom.item())
            self.extras["fseat"] = float((seated.float() * lm).sum().item() / denom.item())
            # best distance achieved this episode (how CLOSE it ever gets) + xy offset
            mdl = self._min_dist < 9.0
            self.extras["fmin"] = float((self._min_dist * mdl.float()).sum().item() / mdl.float().sum().clamp(min=1).item())
            self.extras["fdxy"] = float(((base[:, :2] - goal[:, :2]).norm(dim=-1) * lm).sum().item() / denom.item())
            # force diagnostics: insertion-only (gates break) vs whole-body surf signal
            self.extras["fins"] = float((self._cf_insert * lm).sum().item() / denom.item())
            self.extras["fsurf"] = float((self._cf_filt * lm).sum().item() / denom.item())
            _farm = self._ee_arm_force()
            self.extras["farm"] = float((_farm * lm).sum().item() / denom.item())
            return terminated, truncated

        # ══ Force-signature recovery hooks (RecoveryEnv protocol) ═════════════
        # These let the shared, task-agnostic RecoveryLoop drive this env. They
        # operate on a single instance (env 0) — the recovery demo runs one env.

        @property
        def f_max_n(self) -> float:
            """The per-object force ceiling (LLM budget) for env 0."""
            return float(self._obj_budget[self._obj_cls[0]].item())

        @property
        def max_steps_per_attempt(self) -> int:
            return 240

        def subphase(self) -> str:
            return "insertion"

        def gripper(self) -> str:
            return self.cfg.gripper

        def _sig_window(self, buf: torch.Tensor, w: int) -> torch.Tensor:
            """Last w samples (chronological) for env 0 from a ring buffer."""
            n = min(self._sig_ptr, self._sig_len)
            w = min(w, n)
            if w == 0:
                return torch.zeros(1, device=self.device)
            idx = [(self._sig_ptr - w + i) % self._sig_len for i in range(w)]
            return buf[0, idx]

        def reset_episode(self) -> None:
            self.reset()
            self._rec_off.zero_(); self._rec_steps.zero_(); self._rec_wiggle.zero_()
            self._rec_phase.zero_(); self._jam_cooldown.zero_(); self._rec_open.zero_()
            self._sig_ptr = 0
            self._cf_hist.zero_(); self._basez_hist.zero_()
            self._latx_hist.zero_(); self._laty_hist.zero_()
            self._jam_on[:] = (self.cfg.jam_dx != 0.0 or self.cfg.jam_dy != 0.0)
            self._last_rec_action = None

        def step_skill(self) -> None:
            """One control step under the skill. In forge_mode with a loaded policy,
            the LEARNED policy drives the EE; otherwise zero-action OSC."""
            pol = getattr(self, "_skill_policy", None)
            if pol is not None:
                with torch.no_grad():
                    obs = self._get_observations()["policy"]
                    mean, std = pol(obs, self.f_cmd_norm())
                    # Run the ARM STOCHASTICALLY (sample its action distribution, as the
                    # renderer does). The deterministic mean is shy and stalls against contact
                    # mid-insertion; the on-policy sample pushes through — this is still the
                    # learned policy, just its true (stochastic) form.
                    act = (mean + std * torch.randn_like(std)).clamp(-1.0, 1.0)
                    # ...but GATE the release dim OFF while the loop runs: release is a ONE-WAY
                    # latch on act[7] > 0, and on an object class the policy wasn't trained on
                    # its release head misfires in out-of-distribution states (e.g. mid-air
                    # during the post-recovery re-descent -> the gripper DROPS the bottle). The
                    # loop's success is the geometric seat; release authority belongs to the
                    # caller's finale (which samples the policy's own release distribution once
                    # the bottle is seated). Harness gating, not a scripted release.
                    if self.cfg.forge_release_mode and act.shape[-1] >= 8:
                        act[:, 7] = -1.0
                self.step(act)
            else:
                self.step(torch.zeros(self.num_envs, self.cfg.action_space, device=self.device))

        def _base_xyz0(self):
            o = self.scene.env_origins[0]
            p = self._obj.data.root_pose_w[0]
            return (p[0] - o[0]).item(), (p[1] - o[1]).item(), (p[2] - o[2]).item()

        def is_success(self) -> bool:
            c = self.cfg
            bx, by, bz = self._base_xyz0()
            in_cell = abs(bx - c.rack_x) < 0.04 and abs(by - c.rack_y) < 0.04
            return bool(in_cell and abs(bz - c.cell_floor_z) < c.insert_depth_tol)

        def is_failure(self) -> bool:
            """Jam: high contact with no descent over the window, near the cell."""
            c = self.cfg
            if self._jam_cooldown[0] > 0 or self._rec_steps[0] > 0:
                return False
            if c.forge_mode:
                # forge doesn't advance the phase machine; the "insertion" is active once the
                # scripted setup is done and the bottle is at/near the cell (not still descending
                # from the entrance in free space).
                bx, by, bz = self._base_xyz0()
                # at/around the cell mouth (the wedge sits a little above the floor, and the jam
                # offsets it laterally, so use a generous window) and the scripted setup is done.
                near_cell = (abs(bx - c.rack_x) < 0.09 and abs(by - c.rack_y) < 0.09
                             and bz < c.cell_floor_z + 0.18)
                if int(self._setup_ctr[0]) > 0 or not near_cell:
                    return False
            elif int(self._phase[0]) < int(PickPlacePhase.PLACE_DESCEND):
                return False
            cf_w = self._sig_window(self._cf_hist, c.jam_window)
            bz_w = self._sig_window(self._basez_hist, c.jam_window)
            if cf_w.numel() < c.jam_window:
                return False
            peak = float(cf_w.max().item())
            descent_mm = float((bz_w[0] - bz_w[-1]).item()) * 1000.0   # +ve = went down
            # a wedge = meaningful contact with no descent over the window (caught WELL below
            # break: jam_force_frac keeps the threshold a small fraction of F_max).
            thresh = max(c.jam_force_n, c.jam_force_frac * self.f_max_n)
            return bool(peak >= thresh and descent_mm < c.jam_progress_mm)

        def failure_signature(self):
            from forge_plus.llm.recovery_selector import ForceSignature
            c = self.cfg
            w = c.jam_window
            cf = self._sig_window(self._cf_hist, w)
            bz = self._sig_window(self._basez_hist, w)
            lx = self._sig_window(self._latx_hist, w)
            ly = self._sig_window(self._laty_hist, w)
            dt_ms = float(getattr(self, "step_dt", 1.0 / 60.0)) * 1000.0
            peak_axial = float(cf.max().item())
            mean_axial = float(cf.mean().item())
            net_mm = max(0.0, float((bz[0] - bz[-1]).item()) * 1000.0)
            half = max(1, cf.numel() // 2)
            rising = bool(cf[-half:].mean().item() > cf[:half].mean().item() + 0.2)
            lat_mag = torch.sqrt(lx**2 + ly**2)
            peak_lat = float(lat_mag.max().item())
            mlx, mly = float(lx.mean().item()), float(ly.mean().item())
            bias = "none"
            if (mlx**2 + mly**2) ** 0.5 > 1.0:
                if abs(mlx) >= abs(mly):
                    bias = "+x steady" if mlx > 0 else "-x steady"
                else:
                    bias = "+y steady" if mly > 0 else "-y steady"
            slips = int((lx[1:] * lx[:-1] < 0).sum().item()) if lx.numel() > 1 else 0
            persist = float((cf > 0.5).sum().item()) * dt_ms
            return ForceSignature(
                peak_axial_N=round(peak_axial, 2),
                net_insert_mm=round(net_mm, 2),
                axial_rising=rising,
                lateral_bias=bias,
                contact_persist_ms=round(persist, 1),
                slip_events=slips,
                peak_lateral_N=round(peak_lat, 2),
                mean_axial_N=round(mean_axial, 2),
            )

        def apply_recovery(self, action: str, params: dict) -> None:
            """Execute a recovery primitive on env 0 (F_max never changed here)."""
            c = self.cfg
            d = self.device
            self._last_rec_action = action   # for HUD / logging
            dur = int(params.get("duration_steps", c.rec_dur_steps))
            self._jam_cooldown[0] = dur + c.jam_window
            self._rec_steps[0] = dur
            self._rec_wiggle[0] = False
            self._rec_off[0] = 0.0
            # The recovery maneuver corrects the misaligned approach: clear the
            # induced offset so the re-approach can seat (models real re-alignment).
            self._jam_on[0] = False
            if action == "retract_and_reapproach":
                # lift straight up; on clear, base-aim re-centers and re-descends
                self._rec_off[0, 2] = c.rec_lift
            elif action == "wiggle_search":
                self._rec_off[0, 2] = 0.5 * c.rec_lift
                self._rec_wiggle[0] = True
                self._rec_phase[0] = 0
            elif action == "rotate_align":
                # for a bottle, "align" = nudge the base toward the cell center
                # (counteracts the lateral wedge) while lifting slightly off the rim
                o = self.scene.env_origins[0]
                bx = (self._obj.data.root_pose_w[0, 0] - o[0]).item()
                by = (self._obj.data.root_pose_w[0, 1] - o[1]).item()
                dx, dy = c.rack_x - bx, c.rack_y - by
                nrm = (dx * dx + dy * dy) ** 0.5 + 1e-6
                self._rec_off[0, 0] = (dx / nrm) * c.rec_lat
                self._rec_off[0, 1] = (dy / nrm) * c.rec_lat
                self._rec_off[0, 2] = 0.4 * c.rec_lift
            elif action == "regrasp":
                # re-seat the bottle in the gripper (reuse the warmup seat) + lift
                self._warmup[0] = self.cfg.warmup_substeps
                self._rec_off[0, 2] = 0.4 * c.rec_lift

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
            # reset the per-env-step "released this step" accumulator (set across substeps)
            if self.cfg.forge_release_mode:
                self._newly_released[:] = False

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
            # Insertion-only contact (bottle↔rack) — gates breakage/ceiling in forge.
            _raw_ins = self._raw_insertion_force()
            self._cf_insert = torch.where(
                _live,
                (1.0 - self._cf_alpha) * self._cf_insert + self._cf_alpha * _raw_ins,
                torch.zeros_like(self._cf_insert),
            )
            self._warmup = (self._warmup - 1).clamp(min=0)

            # ── Recovery: force-signature history + maneuver bookkeeping ────────
            base_z_now = self._obj.data.root_pose_w[:, 2] - self.scene.env_origins[:, 2]
            cvec = self._raw_contact_vec3()                       # (N,3) contact force vector
            _p = self._sig_ptr % self._sig_len
            self._cf_hist[:, _p]    = self._cf_filt
            self._basez_hist[:, _p] = base_z_now
            self._latx_hist[:, _p]  = cvec[:, 0]
            self._laty_hist[:, _p]  = cvec[:, 1]
            self._sig_ptr += 1
            self._jam_cooldown = (self._jam_cooldown - 1).clamp(min=0)
            _rec_active = self._rec_steps > 0
            self._rec_steps = (self._rec_steps - 1).clamp(min=0)
            _rec_done = _rec_active & (self._rec_steps == 0)
            if _rec_done.any():
                self._rec_off[_rec_done] = 0.0
                self._rec_wiggle[_rec_done] = False

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
                # induced jam: bias the base-aim off-center so the bottle WEDGES on the cell rim.
                # Gated by self._jam_on: a recovery models a corrected approach -> clears it.
                if self.cfg.jam_dx != 0.0 or self.cfg.jam_dy != 0.0:
                    jam = torch.zeros(self.num_envs, 2, device=self.device)
                    jam[:, 0] = self.cfg.jam_dx
                    jam[:, 1] = self.cfg.jam_dy
                    gate = approach.float() * self._jam_on.float().unsqueeze(-1)
                    p_fixed = torch.cat([p_fixed[:, :2] + gate * jam, p_fixed[:, 2:3]], dim=-1)
            # ── recovery maneuver offset (world frame): lift / lateral search ──
            rec = self._rec_off.clone()
            if self._rec_wiggle.any():
                wig = torch.zeros_like(rec)
                wig[:, 0] = torch.sin(self._rec_phase.float() * 0.5) * self.cfg.rec_lat
                rec = torch.where(self._rec_wiggle.unsqueeze(-1), rec + wig, rec)
                self._rec_phase = torch.where(self._rec_wiggle, self._rec_phase + 1, self._rec_phase)
            p_fixed = p_fixed + rec
            _lam = lam_eff.unsqueeze(-1)
            target_w = ee_pos_w + (p_fixed + a - ee_pos_w).clamp(min=-_lam, max=_lam)
            # ── Soft force ceiling (FORGE force authority): when contact reaches
            # the per-object budget, RETREAT the EE upward a little. This actively
            # bounds the contact force to ~F_max (never near F_break): a wedge then
            # oscillates at the ceiling with no net descent — which the recovery
            # loop catches at LOW force. (Retreat, not freeze, so it can never
            # deadlock once a recovery has cleared the misalignment.)
            if self.cfg.place_strategy == "insert":
                budget = self._obj_budget[self._obj_cls]              # (N,) F_max
                over = self._cf_filt > 0.9 * budget
                back_z = ee_pos_w[:, 2] + 0.004                       # 4 mm up -> relieves contact
                target_w[:, 2] = torch.where(over, back_z, target_w[:, 2])
            # ── FORGE-mode: the LEARNED policy drives the EE (overrides all of the
            # scripted waypoint/base-aim/strategy logic above). Setup-drives to the
            # approach pose first, then the policy's xyz delta does the insertion.
            if self.cfg.forge_mode:
                target_w, ori_k = self._forge_targets(ee_pos_w)
            tgt_pos_b, tgt_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_w, self._ee_quat_des)
            _kpos = torch.full_like(ori_k, 400.0)   # stiff positioning (matches the trained robust policy)
            stiffness = torch.stack([_kpos, _kpos, _kpos, ori_k, ori_k, ori_k], dim=-1)   # (N, 6)
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
            # forge_release_mode: the LEARNED policy commands the release via action[7]>0.
            # Latch it (a one-way commit to the drop) and force the fingers open once set —
            # this overrides the phase-based close so the policy decides WHEN to let go.
            if self.cfg.forge_release_mode:
                rel_cmd = (self._actions[:, 7] > 0.0) & (self._warmup == 0) & (self._setup_ctr == 0)
                # newly_released = the step the policy first lets go (used to penalise releasing
                # while the bottle is still high above the floor -> forces a real descent first).
                # Accumulate across substeps; reset each env step in _pre_physics_step.
                self._newly_released = self._newly_released | (rel_cmd & (~self._released))
                self._released = self._released | rel_cmd
                close_mask = close_mask & (~self._released)
            self._gripper_cmd = torch.where(close_mask.float().bool(), -torch.ones_like(self._gripper_cmd), torch.ones_like(self._gripper_cmd))
            if self.cfg.gripper == "robotiq_2f140":
                # TELEPORT CONTRACT: the 2F-140 is driven by finger_joint position
                # TARGETS only (never write_joint_state — the four-bar loop joints
                # do not survive it). The drive's effort limit (10) bounds the grip
                # force; the release (learned action[7]) is actuated by targeting
                # the open angle — the drive genuinely opens the pads.
                _d = self.device
                ang_t = torch.where(close_mask,
                                    torch.full((self.num_envs,), self._rq_close, device=_d),
                                    torch.full((self.num_envs,), self._rq_open, device=_d))
                ang_t = torch.where(warm, torch.full((self.num_envs,), self._rq_seat, device=_d), ang_t)
                if self.cfg.forge_release_mode:
                    ang_t = torch.where(self._released,
                                        torch.full((self.num_envs,), self._rq_open, device=_d), ang_t)
                r.set_joint_position_target(ang_t.unsqueeze(-1), joint_ids=self._grip_ids)
                eff = torch.zeros(self.num_envs, r.num_joints, device=_d)
                eff[:, :7] = jt
                r.set_joint_effort_target(eff)
                return
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
            # The effort grip CANNOT overcome the near-rigid finger position drive to OPEN — so
            # the "release" target above never actually opens the fingers and the closed gripper
            # keeps carrying the bottle. When the policy has released, drive the fingers open by
            # writing the joint state directly (ramped, gentle), so the gripper genuinely LETS GO.
            # This actuates the LEARNED release decision (action[7]); it is not a scripted skill.
            if self.cfg.forge_release_mode and bool(self._released.any()):
                relm = self._released
                open_w = (0.012 + 0.006 * self._rel_age.float()).clamp(max=0.040)  # ramp open ~5 steps
                fp_w = r.data.joint_pos[:, 7:9].clone()
                fv_w = r.data.joint_vel[:, 7:9].clone()
                fp_w[relm, 0] = open_w[relm]
                fp_w[relm, 1] = open_w[relm]
                fv_w[relm] = 0.0
                r.write_joint_state_to_sim(fp_w, fv_w, joint_ids=[7, 8])

        # ── Observations ──────────────────────────────────────────────────────
        def _get_observations(self) -> dict:
            if self.cfg.forge_mode:
                return self._forge_get_observations()
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
            if self.cfg.forge_mode:
                return self._forge_get_rewards()
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
            if self.cfg.forge_mode:
                return self._forge_get_dones()
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
                bn = list(self._robot.data.body_names)
                self._ee_idx = bn.index("panda_hand")
                if self.cfg.gripper == "robotiq_2f140":
                    self._lf_idx = bn.index("left_inner_finger")
                    self._rf_idx = bn.index("right_inner_finger")
                    jn = list(self._robot.data.joint_names)
                    self._grip_ids = [jn.index("finger_joint")]
                    # Hand the arm to the OSC NOW (parameter write, no body motion):
                    # spawn kept the holding gains so the init sag couldn't tear the
                    # gripper four-bar; from here the OSC owns the arm every step.
                    _z = torch.zeros(self.num_envs, 7, device=self.device)
                    self._robot.write_joint_stiffness_to_sim(_z, joint_ids=self._arm_ids)
                    self._robot.write_joint_damping_to_sim(_z + 80.0, joint_ids=self._arm_ids)
                else:
                    self._lf_idx = bn.index("panda_leftfinger")
                    self._rf_idx = bn.index("panda_rightfinger")
                    self._grip_ids = [7, 8]
            super()._reset_idx(env_ids)
            if self.cfg.gripper == "robotiq_2f140":
                # TELEPORT CONTRACT: no joint-state writes — the four-bar loop joints
                # do not survive them. The demo flow only resets while the arm is still
                # at the (default) spawn pose, and the FORGE setup drive re-poses it.
                jp = self._robot.data.joint_pos[env_ids].clone()
            else:
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
            self._bad_release[env_ids] = False
            self._released[env_ids]    = False
            self._rel_age[env_ids]     = 0
            self._prev_eod[env_ids]    = 0.0
            self._retract_prog[env_ids] = 0.0
            self._succeeded[env_ids]   = False
            self._set_reset[env_ids]   = True
            self._cf_filt[env_ids]     = 0.0
            self._cf_insert[env_ids]   = 0.0
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

            # FORGE mode: a setup window drives the EE to a RANDOMIZED approach pose
            # (the policy must then correct the offset + insert from force).
            if self.cfg.forge_mode:
                n = len(env_ids)
                lat = self.cfg.forge_start_lat
                self._start_off[env_ids] = (torch.rand(n, 2, device=self.device) * 2 - 1) * lat
                self._setup_ctr[env_ids] = self.cfg.forge_setup_steps
                self._settle_ctr[env_ids] = 0
                self._best_dist[env_ids] = 9.9
                self._prev_dist[env_ids] = 9.9
                self._min_dist[env_ids] = 9.9

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
