#!/usr/bin/env python3
"""FrankaGearInsertEnv — FORGE-plus Task 1 gear-on-shaft insertion environment.

Ported from the proven task-3 FrankaGearInsertEnv (branch task3); the machinery
(OSC control, force channels, FORGE learned-insert mode, recovery loop,
Robotiq port) is unchanged — only the task geometry differs. Internal names
kept from the port to minimize the diff: "rack" = the FORGE gear-base plate
(fixed asset), "object"/"bottle" = the held medium gear; "cell" = the middle
shaft the bore goes onto.

Task (issue #26): insert the FORGE medium spur gear (factory_gear_medium.usd,
the exact asset from arXiv 2408.04587 / Isaac Lab GearMesh) onto the middle
shaft of factory_gear_base.usd. Success = bore seated down the 20 mm shaft,
gear resting on the base plate, under the per-object force budget.

Objects (issue #26 F_break distributions on the same gear geometry):
  abs_gear   38±5 N (fragile)    steel_gear  230±20 N (robust)

LLM (Ollama llama3.1:8b) sets F_max per object from IDENTITY ONLY (name +
material — F_break never appears in the prompt: issue-#26 invariant);
cached in llm/budget_cache.json, clamped to [0, 120] N.

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
    from isaaclab.sim import PhysxCfg as _PhysxCfg
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
    #           F_break dist (EVALUATOR ONLY)              grasp (30 mm hub)
    "abs_gear":   {"f_mean":  38.0, "f_std":  5.0, "f_min":  20.0, "grasp_mm": 30.0},
    "steel_gear": {"f_mean": 230.0, "f_std": 20.0, "f_min": 180.0, "grasp_mm": 30.0},
}
OBJ_KEYS  = list(FRAGILE_OBJECTS.keys())   # fixed ordering for tensor indexing
N_OBJ_CLS = len(OBJ_KEYS)


# ─────────────────────────────────────────────────────────────────────────────
# LLM budget helper  (Ollama llama3.1:8b, result cached on disk)
# ─────────────────────────────────────────────────────────────────────────────
_BUDGET_CACHE = "/workspace/FORGE-plus_task3/llm/budget_cache.json"

_MAT_HINT = {
    "abs_gear":   "ABS plastic (thin-walled)",
    "steel_gear": "solid steel",
}


def _llm_query_budget(obj_key: str) -> float:
    """Ask Ollama for a safe F_max (N) for this object. Returns conservative fallback on failure."""
    mat  = _MAT_HINT.get(obj_key, "unknown material")
    mass = {"abs_gear": 12, "steel_gear": 90}.get(obj_key, 30)
    # IDENTITY ONLY — the hidden F_break must never reach the budget-setter
    # (issue #26: "can the frozen LLM infer F_max from object identity alone").
    # Prompt-engineering findings (2026-07-08, llama3.1:8b):
    #   * without the small-parts scale anchor it answers with press-fit
    #     ENGINEERING forces (1400-17000 N);
    #   * free-form replies narrate ("## Step 1...") unpredictably — Ollama's
    #     format=json + temperature 0 makes it terse and deterministic;
    #   * the material-scaling sentence is what differentiates fragile from
    #     robust (without it: 10 N for everything; with it: ABS 10 / steel 100).
    prompt = (
        "A robot arm assembles small machine parts by pressing them onto "
        "shafts (clearance fit; a few Newtons suffice to seat a part). The "
        "force budget caps how hard the robot may ever press: too high risks "
        "cracking or deforming the part, too low wastes headroom for jam "
        "recovery. Scale the budget to the part material: fragile or brittle "
        "materials need LOW budgets, strong ductile metals tolerate HIGH "
        f"budgets. Part: spur gear, 42 mm diameter, made of {mat}, mass "
        f"about {mass} g. The budget must be an integer between 1 and 120 "
        'Newtons. Respond with JSON: {"f_max_n": <integer>}'
    )
    try:
        # HTTP API (stream=false): the CLI subprocess route hit cold-load
        # timeouts and pollutes stdout with TTY control codes.
        import urllib.request as _ur
        req = _ur.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({
                "model": "llama3.1:8b", "stream": False, "format": "json",
                "options": {"num_predict": 80, "temperature": 0},
                "prompt": prompt,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with _ur.urlopen(req, timeout=300) as fh:
            reply = json.loads(json.load(fh).get("response", "{}"))
        v = float(reply["f_max_n"])
        # Validate against the controller-side envelope only ([0, 120] N,
        # issue #26) — NOT against F_break, which the setter must not see.
        return min(max(v, 1.0), 120.0)
    except Exception as exc:
        print(f"[LLM] Ollama failed for {obj_key}: {exc}", flush=True)
    return 15.0   # conservative fixed fallback (identity-blind)


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
class GearInsertEnvCfg(DirectRLEnvCfg if ISAAC_AVAILABLE else object):  # type: ignore[misc]
    # Simulation
    # Factory-matched PhysX config (isaaclab factory_env_cfg): the gear SDF
    # collisions need the big GPU contact/patch buffers and the collision
    # stack — with the task3 defaults, pad<->hub contacts are silently DROPPED
    # for ~85% of envs at 512 envs (probe512: fingers close through the hub,
    # gear falls at warmup; the same envs hold fine at num_envs=4).
    sim: object = SimulationCfg(
        dt=1.0 / 120.0, render_interval=4,
        physx=_PhysxCfg(
            # FULL factory PhysX block. History: this block broke the pad<->hub
            # SDF grip (probe_n4b) — but that was BEFORE the convex hub ring,
            # which makes the grip solver-independent. And the drop test proved
            # the bore is COLLISION-IMPOSSIBLE without factory's contact
            # treatment + SDF fidelity (gear rests flat ON the shaft tip at
            # 0.5 mm offset; 0-offset drop ejects). Factory runs these assets
            # with exactly these settings.
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
    ) if ISAAC_AVAILABLE else None
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
    obj_rest_z:    float = 0.395  # gear origin z resting on table (origin 5 mm below bottom face)
    pre_grasp_z:   float = 0.60   # hover above object before descend
    transport_z:   float = 0.72   # safe altitude for horizontal swing. At radius
                                  # ~0.46 the Franka tops out near z≈0.72, so the old
                                  # 0.80 was unreachable — the hand stalled at ~0.70
                                  # and rammed the object into the (too-tall) rack.
    # Place target: the EE hand height at which the cup's base meets the shelf top.
    # Decoupled from the shelf geometry below. shelf_top 0.50 + cup half 0.045 + the
    # hand->grasp-point offset (~0.067) ≈ 0.61 -> the cup is SET DOWN gently on top.
    place_ee_z:    float = 0.535   # EE hand height at the seated pose: gear origin 0.400 +
                                   # hub grip 0.032 + panda TCP 0.103.
    place_settle_tol: float = 0.02  # |cup_base - shelf_top| to count the cup as placed
    mug_grip_z: float = 0.032  # grip point above gear origin: pad CENTER above the gear top so
                               # the ~24 mm pad's LOWER half grips the Ø35.5 mm hub and its lower
                               # edge (center-12 mm = origin+0.020) clears the Ø41.9 teeth disc
                               # (top at origin+0.015) — at 0.025 the pad bottom overlapped the
                               # teeth band, the warmup teleport interpenetrated and PhysX kicked
                               # the gear out of the pinch
    # Hold the hand TOP-DOWN (approach axis straight down) so a neck-gripped tall object hangs
    # VERTICAL and can be PLACED STANDING. FORGE holds the part at a fixed correct roll/pitch;
    # here that's upright. Needs higher OSC orientation stiffness (the default 40 barely tracks).
    grasp_topdown: bool = True   # top-down hand: a RIGIDLY-gripped gear must hang straight
                                 # under the EE and enter the bore untilted (0.25 mm clearance);
                                 # the bottle's free-hanging pendulum grip did not need this
    # Placement strategy:
    #   "throw_upright" = contact-then-verticalize (preserved old demo: rights the bottle against
    #                     the shelf via a stiffness ramp — looks like flicking it upright).
    #   "extrinsic"     = LEARNED extrinsic dexterity: the policy uses a wrist-pitch action +
    #                     the counter contact to PIVOT the bottle upright, gently. No rigid hold.
    #   "insert" = wine-cellar PEG-IN-HOLE: lower the bottle into a rack cell; the cell walls
    #              align it (contact-rich, FORGE force-guided) and hold it upright. Success = the
    #              base reaches the cell floor (inserted to depth) gently.
    place_strategy: str = "insert"
    rack_z: float = 0.40           # gear-base plate origin z (plate bottom, on the table top)
    cell_floor_z: float = 0.400    # held-gear ORIGIN z when seated on the plate (bottom face 0.405)
    insert_depth_tol: float = 0.006 # |origin - seat| below this counts as fully seated (20 mm travel)
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
    shelf_top_z:   float = 0.425  # middle-shaft TOP (plate 0.40 + shaft height 0.025) = insertion entrance

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
    rec_lat:       float = 0.005   # wiggle/align lateral amplitude (m) — gear scale: the
                                   # bore offset to correct is a few mm (2 cm overshoots 3x)
    rec_dur_steps: int   = 25      # control steps a recovery maneuver runs before clearing
    jam_force_n:   float = 6.0     # contact >= this with no descent => jam (absolute floor)
    jam_force_frac: float = 0.18   # ...or >= this fraction of F_max (whichever is larger). Low
                                   # enough to catch a wedge WELL below break (robust F_max 72 ->
                                   # ~13 N, above clean seating transients, below the ~17 N wedge).
    jam_progress_mm: float = 2.0   # net base descent (mm) over the window below which it's "stuck"
    jam_hover_n:   float = 1.0     # contactless-hover branch: the deterministic action mean can
                                   # back off a slip and hover NEAR the cell forever with ~zero
                                   # contact — no force means the wedge branch never sees it, but
                                   # it is just as stuck. peak BELOW this (with no descent, near
                                   # the cell, above the seat) is also a failure.
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
    forge_approach_z: float = 0.565  # EE hand-off height (nominal; OSC sag delivers the gear
                                     # ~0.46 and the LEARNED policy descends from there — probes
                                     # v3/v4 show it reliably reaches the shaft tip. A scripted
                                     # engagement descent is NOT possible: wrist torque limits
                                     # cap the setup's downward authority at ~10-15 N and it
                                     # stalls 3 cm high — guided probes 1-4).
    forge_start_lat:  float = 0.001  # ± random lateral start offset (m) — sub-mm for the
                                     # engagement staging (stage B widens this back out).
    forge_start_fixed_x: float = -1.0  # >=0: FORCE start_off=(x, y) instead of sampling — jam
    forge_start_fixed_y: float = 0.0   # staging for the funnel probe / jam-recovery eval. The
                                       # setup delivers the gear accurately to shaft+off, so a
                                       # beyond-funnel offset wedges the bore mouth on the shaft.
                                     # Gear bore/shaft radial clearance is ~0.25 mm (true FORGE
                                     # tolerance) — start EASY (5 mm) so success is discovered;
                                     # widen toward 15 mm in follow-up runs (task3 curriculum rule).
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
    release_floor_tol:   float = 0.02   # |origin_z - seat_z| under this counts as resting on the shaft
                                        # (a partially-seated gear sits at most ~20 mm high)
    release_clear_dist:  float = 0.20   # EE must be this far from the bottle BASE (hand RETRACTED
                                        # clear) for success — reached as soon as the moderate retract
                                        # completes (~0.13 is just the grip geometry, so 0.20 = real gap).
    forge_hybrid_retract: bool  = False # after the LEARNED release, drive the hand to a clear pose
                                        # with a fixed retract motion (not a manipulation skill — just
                                        # clearing out so the let-go is visible). Release stays learned.

    # ── Table-pick staging (robotiq 2F-140) ─────────────────────────────────────
    rq_table_pick: bool = False   # gear spawns RESTING on the table at a pick spot
                                  # and the staged close grips the SUPPORTED part —
                                  # no seat-window pin (the in-air pin teleports
                                  # were the last non-physical assist). The OSC
                                  # then lifts and carries to the same trained
                                  # hand-off entrance (the bottle-era carry leg).
    rq_pick_dth: float = -0.33    # pick spot = entrance xy rotated by this angle
                                  # about the robot base, so the pick arm pose is
                                  # the PROVEN grasp pose with j1 offset by the
                                  # same angle (no new pose engineering; the gear
                                  # is rotationally symmetric so pad yaw is free)

    # ── Payload compensation (robotiq): feedforward J^T gravity wrench ─────────
    # PhysX's articulation gravity comp is short for the loop-jointed 2F-140
    # subtree and knows nothing about the held bottle. The un-modeled gravity
    # MOMENT of the long loaded tool is what pinned the wrist at its 12 Nm
    # clamp (the lean/twist/grind in every robotiq take). These scale the
    # feedforward of the measured subtree/bottle gravity wrench through J^T.
    # 0.0 = off. Calibrated from the RQ_TRACE3 static-hover wrench fit.
    # (Tried 1.0/1.0 against the abs descent drift: the arm FLOATS — PhysX
    # already models most of the gripper gravity, so the full subtree wrench
    # double-compensates ~6 N of lift and contact is never reached. The
    # missing wrench is only the loop-joint residual; leave pc OFF.)
    rq_pc_grip: float = 0.0   # scale on the robotiq-subtree gravity wrench
    rq_pc_obj:  float = 0.0   # scale on the held-object gravity wrench
    rq_pc_moment: float = 1.0  # MOMENT-ONLY payload comp (force rows zero): the
                               # ag integrator owns the lift (full-wrench comp
                               # double-counted it and floated the arm), but the
                               # un-modeled gripper-subtree gravity MOMENT is the
                               # constant lateral push that walked every free
                               # hover to a ~35 mm offset equilibrium (probe 9
                               # case B: all 8 envs identical to 0.3 mm).

    forge_no_term:    bool  = False  # render-only: never auto-terminate (so the seated bottle isn't
                                     # reset away before the camera captures the release/retract)
    render_minimal:   bool  = False  # render-only: skip the filtered insert-sensor + compliant-material
                                     # binding (extra SDG-graph state not needed to capture frames)
    forge_obj_cls:    int   = 1      # fix the training object (1=steel_gear, robust) so breakage does
                                     # not derail learning the SEAT; -1 = randomize. Fragility curriculum
                                     # (transfer to glass) is a follow-up once seating is learned.

    # ── Jam induction: in-grip slip disturbance (issue #26 recovery eval) ────
    # One-time lateral teleport of the gear INSIDE the pinch once it descends
    # into the funnel region: models a grip slip. The policy (no vision, ft
    # only) keeps pressing "aligned" while the bore rim sits on the shaft tip
    # -> a REAL sustained wedge with an insertion-force signature. (A static
    # start offset cannot wedge the trained policy: <=13 mm it self-corrects,
    # >13 mm it wanders off and parks force-free — funnel probe + debug ep0.)
    slip_disturb_mm: float = float(os.environ.get("SLIP_DISTURB_MM", "0.0"))
                                     # 0 = off; sign = +y. Env-var knob so TRAINING can put
                                     # jam states in-distribution without code edits.
    slip_rand_dir:   bool  = bool(int(os.environ.get("SLIP_RAND_DIR", "0")))
                                     # randomize the slip direction (uniform xy angle) instead
                                     # of the fixed +y. TRAINING must set this: with a fixed
                                     # direction the policy learns to AIM OFF-CENTER in
                                     # anticipation (slipfrac run: jam eval 6/6 but clean eval
                                     # 0/256 — the slip "corrected" the learned bias). A zero-
                                     # mean disturbance leaves reactive recovery as the only
                                     # winning strategy. Eval keeps the deterministic +y.
    slip_frac:       float = float(os.environ.get("SLIP_FRAC", "1.0"))
                                     # fraction of episodes the slip fires in (per-env random
                                     # arm rolled at reset). Slip-EVERY-episode training failed
                                     # (275 its, zero seats — the 5 mm kick starves the clean-
                                     # skill success signal); ~0.3 keeps clean insertions
                                     # paying while recovery states enter the distribution.
                                     # Eval keeps 1.0 (every episode jams).
    slip_trigger_z:  float = 0.435   # gear origin below this (shaft-tip region) arms the slip

    # ── Budget-setter baselines (issue #26 eval) ─────────────────────────────
    budget_mode:    str   = "llm"    # llm | fixed | oracle
    budget_fixed_n: float = 60.0     # fixed-global F_max (120 = no-ceiling)
    oracle_margin_n: float = 5.0     # oracle: F_max = F_break - margin

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
class MockGearInsertEnv(gym.Env):
    """Lightweight CPU mock of FrankaGearInsertEnv.

    No Isaac Sim required.  Joint kinematics are approximated; contact forces
    are simulated from the phase.  Use this to verify obs shapes, reward logic,
    and training loop plumbing before GPU runs.
    """

    metadata = {"render_modes": []}

    def __init__(self, num_envs: int = 4, device: str = "cpu"):
        super().__init__()
        self.num_envs = num_envs
        self.device   = torch.device(device)
        self.cfg      = GearInsertEnvCfg()
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

    class FrankaGearInsertEnv(DirectRLEnv):
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

        cfg: GearInsertEnvCfg
        NUM_PHASES = NUM_PHASES

        def __init__(self, cfg: GearInsertEnvCfg, render_mode=None, **kw):
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
                # GEAR PORT: the robotiq needs the clamp OPEN — every ori_k
                # command above 600 (the "1200" free-space stiffness) was
                # silently clamped here, so the wrist ran at HALF the assumed
                # gain and its tilt equilibrium against the ~2-3.6 Nm hanging
                # moment sat at ~12 deg (probe 5: gripswing 52 mm, perfectly
                # aligned hover that could never descend — a 12 deg-cocked
                # bore cannot drop over the shaft).
                motion_stiffness_limits_task=(
                    (5.0, 3000.0) if self.cfg.gripper == "robotiq_2f140"
                    else (5.0, 600.0)),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
                motion_stiffness_task=[400.0, 400.0, 400.0, 40.0, 40.0, 40.0],  # initial; overridden per step
                motion_damping_ratio_task=1.0,                  # critically damped
                motion_control_axes_task=[1, 1, 1, 1, 1, 1],
                nullspace_control="position",
                # GEAR PORT robotiq: 150 (was 15). The 2F-140's SINGLE actuated
                # knuckle (finger_joint) at pinch stall reacts ~2.3 Nm onto the
                # gripper chassis with no mirrored drive to cancel it; in
                # effort mode only this nullspace spring resists the resulting
                # self-motion, and at 15 the equilibrium is 2.3/15 = 0.153 rad
                # of j5 wind (hold-trace 3: j5 +0.12 rad and climbing = the
                # entire post-hand-off walk). 150 caps it at ~0.9 deg. Posture
                # torque is task-projected, so tracking is unaffected.
                nullspace_stiffness=(150.0 if self.cfg.gripper == "robotiq_2f140"
                                     else 15.0),
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
            self._budget_env = torch.zeros(N, device=d)   # per-episode F_max (budget_mode)
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
            # robotiq seat WINDOW: the arm spawns folded (no teleport under the
            # contract) and the OSC traverse pace varies, so the seat is ARMED BY
            # ARRIVAL at the stage-2 hand-off pose (not by a countdown value) and
            # runs on its own counter while the arm freezes and setup_ctr pauses.
            # _rq_seatctr: -1 = pending (traverse; bottle parked), >0 = seat active
            # (probe-v50 cadence below), 0 = done (grip established, setup resumes).
            self._rq_seatctr = torch.zeros(N, dtype=torch.long, device=d)
            # PRE-DRIVE (robotiq): joint-position-drive the arm from the folded USD
            # spawn to the probe-approved side-grip pose on the HOLDING gains (the
            # probe's servo mechanism — proven tear-free; asset NOTE 4: baking the
            # pose as spawn state tears the four-bar at parse). The OSC gets the
            # arm only when this counter expires (gains zeroed at that moment).
            self._rq_predrive = torch.full((N,), -1, dtype=torch.long, device=d)
            self._RQ_PREDRIVE = 120   # GEAR PORT: the reset ARM-ONLY teleport puts
                                      # the arm AT the grasp pose, so the predrive
                                      # only settles the drives (bottle: 600 to
                                      # unfold from the factory spawn)
            if cfg.rq_table_pick:
                # table pick needs REAL servo windows (%120 slots + the pd==120
                # tolerance check are unreachable at PREDRIVE=120 — probe 2:
                # zero windows ran, seat armed 4 cm high): the z leg walks the
                # pads from the teleport altitude down to the grip band over
                # the resting gear.
                self._RQ_PREDRIVE = 600
            self._rq_pd_ext = 0       # tolerance-gated extensions used (max 4 x 240)
            self._rq_drive_kp = None  # holding-gain snapshot, taken just before the
            self._rq_drive_kd = None  # OSC handoff zeroes the arm drives (restored
                                      # at each full reset so ep-2+ predrive works)
            # The closed-loop pose servo is OFF: chasing the hand-off pose walks
            # the arm into OSC-hostile pose families (smoke 13 vertical-shoulder
            # runaway, smoke 14 reared-up clamp corner). The open-loop command
            # reproduces the probe pose exactly (smoke 9: ee (0.154,0,0.75),
            # level) — the seat happens THERE, and the OSC carries the gripped
            # bottle to the cell (carry stability proven in smokes 12/13).
            self._rq_servo_on = False
            if cfg.rq_table_pick:
                # TABLE PICK: the open-loop command holds the IN-AIR hand-off
                # altitude — the pads closed 3.8 cm above the hub, around the
                # bore cylinder, to full-close (probe 1). The servo's table-mode
                # branch retargets z to the grip band over the resting gear;
                # its clamped per-window trims start from an already-correct
                # pose family (same pose, j1 offset), so the pose-family
                # walk-off that got the servo disabled cannot engage.
                self._rq_servo_on = True
            # Integral z feedforward for the OSC carry: PhysX's gravity comp is
            # ~50 N-equivalent short for this loop-jointed articulation in
            # extended poses (smoke 15: 13 cm sag drove the carried bottle into
            # the rack wall at 25 N). The integrator self-calibrates the missing
            # lift during the scripted setup and FREEZES when the policy goes
            # live (so it never fights the intentional insertion contact).
            self._rq_fup = torch.zeros(N, device=d)   # (legacy J-row lift — unused)
            self._rq_ag  = torch.zeros(N, device=d)   # adaptive gravity-comp scale (tau += ag*grav)
            self._rq_ag_cap = None    # post-recovery ag ceiling (rec_end snapshot;
                                      # downward-only adaptation while centered)
            # Payload compensation (cfg.rq_pc_grip / rq_pc_obj): lazily-resolved
            # gripper-subtree body ids + masses and the object mass (PhysX truth).
            self._pc_grip_ids: list[int] | None = None
            self._pc_grip_m: torch.Tensor | None = None
            self._pc_obj_m: torch.Tensor | None = None
            self._rq_f   = torch.zeros(N, 3, device=d)  # LEARNED-phase residual EE force (N)
            self._rq_prev_ee: torch.Tensor | None = None   # realized-vs-commanded step memory
            self._rq_prev_cmd = torch.zeros(N, 3, device=d)
            self._rq_carry_dz = -0.137  # GEAR PORT: carry EE == seat EE (0.692) so
                                      # the CARRY stage only — transport_z + 0.111
                                      # would put the EE at 0.83, past the arm's
                                      # ~0.72 top-down ceiling at this radius. At
                                      # 0.72 the gear traverses at ~0.474, 5 cm
                                      # above the shaft tops (0.425).
                                      # (bottle history: +0.13, smokes 28/29)
            self._rq_des_z = torch.zeros(N, device=d)  # unclamped setup z target (integrator error)
            self._rq_xyint = torch.zeros(N, 2, device=d)  # table-pick XY standing-error
                                      # integrator (the pin used to hide the OSC's
                                      # ~12 mm appr offset from the arrival gate)
            self._rq_lift_ok = True   # False between OSC handoff and lift convergence:
            self._rq_liftctr = 0      # xy HELD at the handoff anchor until the integrator
                                      # has learned the lift (carrying early = rack strike)
            self._rq_hover_xy = torch.zeros(N, 2, device=d)  # FIXED hover anchor — following
                                      # the live xy is a moving target = zero xy stiffness
                                      # and the arm drifts away unopposed (smoke 18)
            # Grip height on the bottle (robotiq): 0.14, NOT cfg.mug_grip_z 0.12.
            # Mesh profile (scale 0.5): the neck TAPERS up to z~0.115 (16.6->18.5
            # mm below it — a camming ramp that converts any axial load into a
            # pad-opening force; the bottle slid up through the pinch in every
            # press/wedge, smokes 30-37). 0.12-0.15 is a TRUE 15.5 mm cylinder
            # with the string-ring bulge at 0.142 — gripping at 0.14 puts the pad
            # column on the cylinder with the RING inside the pinch: a positive
            # interlock against axial slip.
            # GEAR PORT: pad centre at gear origin +0.040 — the 2F-140's ~40 mm
            # pads need the lower edge (centre - 20 mm) to clear the Ø41.9 teeth
            # disc (top at origin +0.015); 0.040 leaves 5 mm and still spans the
            # upper 25 mm of the Ø35.5 hub (15->45 mm). The franka's 0.032 with
            # 24 mm pads has the same 5 mm tooth clearance.
            # (bottle value was 0.14 — the string-ring interlock band.)
            self._rq_grip_h = 0.030
            self._rq_rimz = torch.full((N,), -1.0, device=d)  # rim-contact altitude (-1 unset)
            self._rq_pressctr = torch.zeros(N, dtype=torch.long, device=d)  # sustained-press
                                      # substeps > 1.5 N during a centered descent (stall
                                      # detector for the rim gate — see the gate comment)
            self._rq_aim = torch.zeros(N, 2, device=d)  # FROZEN post-recovery ee aim
                                      # (see the rec_end block: live base-aim feedback
                                      # pumps the held bottle's pendulum swing)
            self._rq_rec_prev = torch.zeros(N, dtype=torch.long, device=d)  # recovery edge detect
            self._rq_regrasp_pend = torch.zeros(N, dtype=torch.bool, device=d)  # regrasp seat
                                      # DEFERRED to the maneuver falling edge: the seat
                                      # window PINS the gear to the shaft axis at
                                      # grasp_c_z - grip_h, so firing it while the gear
                                      # is pressed in contact (the abs post-retract
                                      # friction press) teleports it into overlap with
                                      # the shaft — PhysX penetration spikes to ~300 N
                                      # and breaks fragile gears (rq sweep ours cell:
                                      # 8/25 breaks). The lift clears contact first.
            self._rq_reaim_pend = torch.zeros(N, dtype=torch.bool, device=d)  # re-freeze the
                                      # post-recovery aim AFTER the deferred seat: the
                                      # rec_end freeze sees the PRE-seat gear pose, and
                                      # the seat then re-centers the gear in-hand — a
                                      # stale aim lands the re-descent off-center
                                      # (+y 30 N wedges, absfix smoke)
            self._rq_seat_hold = torch.zeros(N, 3, device=d)  # FIXED arm hold during the
                                      # seat window: holding at the LIVE ee is zero-
                                      # error = zero restoring force (take-97 rule) —
                                      # under OSC (recovery regrasp) the arm crept 2 cm
                                      # through the squeeze and the bite failed at unpin
                                      # (absfix smoke ep0: gear dropped, tilt 97°)
            self._rq_pendctr = torch.zeros(N, dtype=torch.long, device=d)  # pend age /
                                      # fire-anyway cap for the deferred regrasp seat
            self._rq_calmctr = torch.zeros(N, dtype=torch.long, device=d)  # swing-settle gate:
                                      # consecutive substeps with the held bottle's
                                      # |v_xy| calm; the post-recovery re-descent
                                      # WAITS for the maneuver's pendulum swing to
                                      # decay (descending while swinging PUMPS it
                                      # under render tracking — take 98/99 flings)
            self._rq_need_settle = torch.zeros(N, dtype=torch.bool, device=d)
                                      # settle REQUEST latch, set at every rec_end.
                                      # (The gate cannot key off ~_rq_desc: the descent
                                      # latch deliberately stays True through recovery —
                                      # see the rec_end block — so a ~_rq_desc term made
                                      # the gate structurally dead post-recovery, take 97.)
            self._rq_settle_pos = torch.zeros(N, 3, device=d)
                                      # FIXED hold point while settling (snapshotted at
                                      # rec_end). Holding at the LIVE ee gives zero
                                      # restoring force — the maneuver's residual arm
                                      # momentum dragged the held bottle 30+ cm (take 97).
            self._rq_settlewait = torch.zeros(N, dtype=torch.long, device=d)
                                      # total substeps spent settling this recovery —
                                      # hard cap so a bottle that never reads calm under
                                      # render physics can't hold the demo to timeout
            self._rq_centered = False  # True after the first recovery centers the approach
            self._rq_desc = torch.zeros(N, dtype=torch.bool, device=d)  # LATCHED descent:
                                      # a hysteresis-free arrival gate chattered the z
                                      # target 0.80/0.68 for 400 substeps at the boundary
                                      # and destabilized the hover (smoke 21)
            # COMMAND targets, droop-compensated (= 2*desired - reached, smoke 8):
            # the holding gains sag under gravity, so commanding the desired pose
            # directly lands j2/j4 ~0.2 rad low (hand pitched ~23 deg down).
            # Desired MEASURED pose: [0, -1.4089, 0, -2.6779, 0, 2.896, 0.7204].
            # _rq_arm_cmd is the LIVE command the predrive SERVO trims every 120
            # substeps (probe servo: z err -> j2, pitch err -> j6) so the hand ends
            # LEVEL at the hand-off height regardless of droop.
            # GEAR PORT: the Newton-converged droop-compensated COMMAND that
            # holds the top-down grasp pose on the holding gains (probe run 3:
            # reached ee (0.288, 0, 0.726), pitch 0.03 deg).
            self._rq_arm_pose = torch.tensor(
                [0.2591, -0.4489, 0.0, -1.4194, 0.0, 1.4375, 0.785], device=d)
            self._rq_arm_pose_ent = self._rq_arm_pose.clone()  # entrance pose —
                                      # the drive-side return traverse target
            self._rq_return = -1      # substep countdown of that traverse
                                      # (-1 = not started this episode)
            if cfg.rq_table_pick:
                # table pick: same pose family, base yaw offset to the pick spot
                self._rq_arm_pose[0] += cfg.rq_pick_dth
            self._rq_arm_cmd = self._rq_arm_pose.clone()
            self._rq_drive_mode = False  # True (robotiq) until the seat completes:
                                         # arm on HOLDING-GAIN joint drives (probe
                                         # regime: zero sag, level pads); the OSC
                                         # gets the arm only after the grip holds.
            self._RQ_SEAT_HI = 270   # window length: pin+settle 50, kiss 50, ...
            self._RQ_KISS_HI = 220   # ctr<=220: pads target kiss
            self._RQ_KISS_LO = 170   # ctr<=170: full-close squeeze (0.785)
            self._RQ_PIN_LO  = 40    # ctr<=40: unpin. The squeeze must be FULLY
                                     # stalled+settled before release: unpinning at
                                     # 110 freed the bottle mid-close and the still-
                                     # closing pads extruded it down the neck taper
                                     # (smokes 9/11: pinned stall 0.587 = correct,
                                     # post-unpin creep to 0.70 = escape).
            # Distance from the hand origin to the grasp point (between the fingertip
            # pads) along the hand's local +z. Per-env so it can be swept/calibrated;
            # 0.067 seats the block centrally between the panda pads (calibrated).
            # The Robotiq 2F-140's pads sit much lower below the flange (calibrated
            # empirically via the headless recovery probe).
            # robotiq: the grip point (neck between the fingertip pads) sits 0.214 m
            # along the hand's local +z from robotiq_base_link (probe v50: EE at
            # x=0.145, held grip at x=0.359, approach = local +z). Same PERPENDICULAR
            # side grip as the franka demo, just a longer hand.
            # franka: 0.103 = flange->fingertip-pad-centre (standard panda TCP).
            # (task3 used 0.067 — calibrated for the bottle-NECK grip higher up
            # the pads; with the gear it seated the hub BELOW the pads and the
            # fingers closed on air -> the gear dropped and flipped at warmup.)
            _tcp = 0.214 if self.cfg.gripper == "robotiq_2f140" else 0.103
            self._grasp_tcp_d = torch.full((N,), _tcp, device=d)
            # Gripper-length delta vs the franka hand: shifts the FORGE hand-off /
            # transport / retract HEIGHTS so the BOTTLE traverses the same altitudes.
            # With the perpendicular side grip the extra length extends HORIZONTALLY,
            # so no height shift (the old 0.25 was for the abandoned top-down lip grip).
            # -0.045 (robotiq): the horizontal side grip holds the bottle base at
            # grasp_z - 0.12 = 0.535 at the franka's approach altitude — 4 cm of
            # FREE SPACE above the rim. The policy was trained ENTIRELY in-contact
            # (franka's tilted grasp parks the base ON the rim, cf ~6 N from step
            # one); in free space its action mean saturates (+1,+1) and it flees
            # (offline probe, smoke 27). Lowering the hand-off by 4.5 cm restores
            # the trained bottle-on-rim contact state at the policy hand-off.
            # GEAR PORT: the grip is TOP-DOWN (same as the franka — the rigidly
            # held gear must hang straight to thread the 0.4 mm bore clearance),
            # so the extra hand length extends VERTICALLY: EE heights shift UP by
            # (0.214 + 0.040) - (0.103 + 0.032) = +0.119 for the gear to traverse
            # the franka's altitudes. (bottle: -0.045, horizontal side grip.)
            self._tcp_dz = 0.109 if self.cfg.gripper == "robotiq_2f140" else 0.0
            # The FORGE setup dest places the EE at the cell xy; with the horizontal
            # side grip the BOTTLE then hangs one tcp-length further along the +x
            # approach axis. The learned policy was trained on the franka's ~6.7 cm
            # lean — keep the BOTTLE at that same offset by pulling the (longer)
            # robotiq EE dest back by the tcp difference.
            # GEAR PORT: 0 — the top-down grip hangs the gear directly under the
            # EE (no horizontal tcp offset to compensate), and the gear task
            # seeds jams with the IN-GRIP SLIP inducer, not the bottle's wedge
            # staging (which needed dest_dx = jam_dx - tcp for the side grip).
            self._rq_dest_dx = 0.0
            # Finger half-opening to rest the pads at during warmup: hub half-width
            # (0.01775) minus ~1 mm so the pads sit just at the surface (clean seat,
            # no deep penetration), then the PD grip takes over and holds by friction.
            self._grasp_seat_w = 0.0168
            # Robotiq finger_joint targets (rad; 0 = OPEN, 0.785 = full close — the
            # STANDARD convention; the old "inverted" reading was an artifact of the
            # four-bar assembling in the wrong branch, fixed by the outer_finger lock
            # baked into the asset — build_franka_robotiq_2f140.py NOTE 3).
            # Calibrated on the fixed mechanism (probe v50, scale-0.5 bottle, 16 mm
            # neck): tip gap 59 mm at 0 -> 0 mm at ~0.35; KISS at 0.218 (24 mm gap);
            # SQUEEZE = full-close command 0.785 — the drive stalls on the neck at its
            # effort limit (ang ~0.589, bounded pinch), pads parallel throughout.
            self._rq_close  = 0.785
            # NOTE (smokes 51/52): do NOT overdrive the hold target past 0.785 to
            # strengthen the pinch — the four-bar pads tilt under the higher squeeze
            # and axially SQUIRT the bottle out (stepped 1.2: instant ejection at
            # handoff; ramped 0.95: ejection as the ramp deepened the stall to ~0.727).
            # The 0.785 hold at the ~0.705 neck stall is the calibrated stable grip.
            # GEAR PORT: kiss just short of hub contact (probe run 3: first
            # contact at ang 0.607, full-close stall 0.6165 at 19.4 N).
            # (bottle: 0.218 — 24 mm gap on the 16 mm neck.)
            self._rq_seat   = 0.52
            self._rq_open   = 0.0
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
            self._fk_aim     = torch.zeros(N, 2, device=d)  # franka post-recovery frozen gear-aim
            self._jam_on     = torch.zeros(N, dtype=torch.bool, device=d)  # induced misalignment active
            self._last_rec_action = None     # last recovery primitive applied (HUD)

            # ── FORGE-mode state (learned insertion) ──
            self._skill_policy = None     # optional loaded policy that drives step_skill (forge)
            self._setup_ctr  = torch.zeros(N, dtype=torch.long, device=d)  # approach-pose setup countdown
            self._start_off  = torch.zeros(N, 2, device=d)                 # random lateral start offset
            # ANCHORED policy target (2F-140 plant fix — see _forge_targets)
            self._pol_anchor     = torch.zeros(N, 3, device=d)
            self._pol_anchor_set = torch.zeros(N, dtype=torch.bool, device=d)
            self._best_dist  = torch.full((N,), 9.9, device=d)             # best (min) dist-to-goal this episode
            self._prev_dist  = torch.full((N,), 9.9, device=d)             # PBRS: previous-step dist-to-goal
            self._prev_dxy   = torch.full((N,), 9.9, device=d)             # PBRS: previous-step xy offset
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
            raw     = [budgets.get(k, 15.0) for k in OBJ_KEYS]  # identity-blind fallback
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

            # (recipe v3 needs no parse ghost — the NVIDIA-pattern attachment
            # materializes the four-bar loop joints on its own.)
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
                # GEAR PORT: base at the ORIGIN, franka-identical — the gear grip
                # is top-down (not the bottle's horizontal side grip), so the
                # scene geometry (reach to the shaft at (0.45, 0.12), radius
                # ~0.466) should match the franka's exactly. The bottle's
                # (-0.15, +0.12) offset lengthened the reach to 0.60 m, which the
                # top-down EE ceiling (~0.72 at this radius) cannot afford.
                robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
                _jp = {k: v for k, v in robot_cfg.init_state.joint_pos.items()
                       if not k.startswith("panda_finger")}
                _jp.update({"finger_joint": 0.0, ".*_inner_finger_joint": 0.0,
                            ".*_inner_finger_pad_joint": 0.0, ".*_outer_.*_joint": 0.0})
                # GEAR PORT: default = the REACHED top-down grasp pose from the
                # Newton-converged calibration (probe_rq_gear_seat run 3): hand
                # straight down, robotiq_base_link at (0.288, 0, 0.726), grasp
                # point at 0.512, gear seat at 0.472. _reset_idx's ARM-ONLY
                # teleport puts the arm here every episode; the droop-compensated
                # predrive command (_rq_arm_pose) holds it on the stiff drives.
                _jp.update({"panda_joint1": 0.259, "panda_joint2": -0.2282,
                            "panda_joint3": 0.0, "panda_joint4": -1.6545,
                            "panda_joint5": 0.0021, "panda_joint6": 1.4281,
                            "panda_joint7": 0.7853})
                robot_cfg.init_state.joint_pos = _jp
                # Actuators: isaaclab's UR10e+2F-140 template. Drive the finger_joint;
                # the pad springs adapt; the rest is loop-/mimic-owned (passive).
                from isaaclab.actuators import ImplicitActuatorCfg
                robot_cfg.actuators.pop("panda_hand", None)
                robot_cfg.actuators["gripper_drive"] = ImplicitActuatorCfg(
                    # GEAR PORT: effort 60 (bottle: 30 — its axial-slip story,
                    # smokes 33/34). The pinch's ANTI-TILT moment scales with
                    # the squeeze: at 30 (~24 N pinch) a ~10 N off-axis rim
                    # press ROTATES the gear in-grip (abs debug smoke: tilt
                    # 3.4->6.6 deg while pressing -> wedge -> timeout); a
                    # tilted bore cannot thread the 0.4 mm fit band. The
                    # franka's 74 N pinch holds the same press rigid.
                    # Pinch force = drive STIFFNESS x (close-target - stall)
                    # (~2.3 Nm -> ~24 N at 11.25; the effort limit never
                    # binds). 22.5 was tried against the abs press-tilt and
                    # made contact-keeping WORSE — the tilt driver is the
                    # WRIST (weak OSC ori channel), not the pinch; keep the
                    # kiss/stall-calibrated 11.25.
                    # SYMMETRIC drive (hold-traces 3/4): a single actuated
                    # knuckle at pinch stall reacts ~2.3 Nm onto the chassis
                    # with nothing to cancel it — in OSC effort mode that
                    # torque winds j5 to the ori-gain equilibrium (~0.15 rad,
                    # the entire post-hand-off walk; nullspace is projected
                    # out since the wind moves the task). Driving BOTH
                    # knuckles at the mirrored target cancels the reactions
                    # on the chassis (like the real gripper's gear train).
                    # 5.625 each keeps the calibrated 11.25 total -> same
                    # kiss/stall/pinch.
                    joint_names_expr=["finger_joint", "right_outer_knuckle_joint"],
                    effort_limit_sim=60.0,
                    velocity_limit_sim=1.0, stiffness=5.625, damping=0.1,
                    friction=0.0, armature=0.0)
                # Follower joints must be (near-)undriven: the LIVE four-bar loop
                # owns their kinematics (inner_finger = -finger_joint exactly, pads
                # parallel). Any real stiffness here fights the loop (probe v48-50).
                robot_cfg.actuators["gripper_finger"] = ImplicitActuatorCfg(
                    joint_names_expr=[".*_inner_finger_joint"], effort_limit_sim=0.01,
                    velocity_limit_sim=1.0, stiffness=0.0, damping=0.001,
                    friction=0.0, armature=0.0)
                robot_cfg.actuators["gripper_passive"] = ImplicitActuatorCfg(
                    joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint"],
                    effort_limit_sim=1.0, velocity_limit_sim=2.0, stiffness=0.0,
                    damping=0.0, friction=0.0, armature=0.0)
            else:
                robot_cfg.spawn.usd_path = "/workspace/assets/franka/panda_instanceable.usd"
                # GEAR PORT: spawn at the CLASSIC top-down ready pose (hand
                # pointing STRAIGHT DOWN at ~(0.31, 0, 0.59)). Neither task3's
                # side-grip overrides (-0.73/-2.46/2.85, handdn 0.44) nor the
                # isaaclab_assets default (a forward-tilted reach pose, handdn
                # 0.64) start vertical, and the J^T OSC cannot close tens of
                # degrees of orientation error (probes: gear carried at 44 deg
                # tilt / arm flail + dropped gear). The topdown grasp must START
                # at the commanded orientation.
                _jp = dict(robot_cfg.init_state.joint_pos)
                _jp.update({"panda_joint1": 0.0, "panda_joint2": -0.785,
                            "panda_joint3": 0.0, "panda_joint4": -2.356,
                            "panda_joint5": 0.0, "panda_joint6": 1.571,
                            "panda_joint7": 0.785})
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
                        # The FORGE medium spur gear (exact GearMesh asset, arXiv
                        # 2408.04587): Ø42 mm teeth, 30 mm cylindrical hub gripped by
                        # the pads, through-bore for the shaft. Origin 5 mm below the
                        # bottom face; the seat offsets by mug_grip_z to grip the hub.
                        usd_path="/workspace/assets/factory/gear_medium_body.usda",
                        scale=(1.0, 1.0, 1.0),
                        # factory USDs lack the contact-reporter API the surf/insert
                        # sensors need (LIBERO assets had it baked in)
                        activate_contact_sensors=True,
                        # Nominal FORGE mass is 12 g; 30 g keeps the friction grip and
                        # jam force-signature well-conditioned for BOTH material classes
                        # (class only changes the hidden F_break, as in task3).
                        mass_props=sim_utils.MassPropertiesCfg(mass=0.030),
                        # DYNAMIC — real physics: held by the friction grip, carried, released.
                        # Factory-matched contact params (factory_tasks_cfg gear spawns +
                        # factory_env_cfg "important to avoid interpenetration"): high
                        # solver iterations + explicit contact offset for the SDF mesh.
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            kinematic_enabled=False, disable_gravity=False,
                            max_depenetration_velocity=1.0,
                            solver_position_iteration_count=192,
                            solver_velocity_iteration_count=1,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.005, rest_offset=0.0,
                        ),
                        # The factory USD has PhysicsArticulationRootAPI on the body
                        # (factory spawns gears as Articulations); disable it so the
                        # prim resolves as a plain rigid body.
                        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                            articulation_enabled=False,
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(self.cfg.rack_x, self.cfg.rack_y, self.cfg.transport_z),
                    ),
                )
            )

            # FORGE gear-base plate ("rack" in ported code): three shafts on a
            # 150x75 mm plate. The MIDDLE shaft (x +0.02025 in the plate frame) is
            # the insertion target, so the plate spawns offset -0.02025 in x and
            # the shaft center lands exactly at (rack_x, rack_y). Shaft top at
            # rack_z + 0.025 = shelf_top_z.
            self._rack = RigidObject(
                RigidObjectCfg(
                    prim_path="/World/envs/env_.*/Rack",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path="/workspace/assets/factory/gear_base_body.usda",
                        activate_contact_sensors=True,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(
                            contact_offset=0.005, rest_offset=0.0,
                        ),
                    ),  # compliant contact material applied post-spawn (UsdFileCfg rejects it)
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(self.cfg.rack_x - 2.025e-2, self.cfg.rack_y, self.cfg.rack_z),
                    ),
                )
            )

            # ── Sanitize the factory gear-base USD (env_0, pre-clone) ──────────
            # factory_gear_base.usd carries a PhysicsFixedJoint `root_joint` with
            # PhysicsArticulationRootAPI (factory's fixed-asset articulation
            # pattern). As a KINEMATIC RigidObject that joint is static↔static
            # (PhysX CreateJoint error) and the articulation root confuses the
            # rigid-body resolve — deactivate the joint prim outright.
            import omni.usd as _ousd
            _stage = _ousd.get_context().get_stage()
            # Body-root wrappers put the rigid bodies AT /Object and /Rack; make
            # sure both carry the contact-report API (idempotent).
            from pxr import PhysxSchema as _PhysxSchema
            for _bp in ("/World/envs/env_0/Object",
                        "/World/envs/env_0/Rack"):
                _prim = _stage.GetPrimAtPath(_bp)
                if _prim and _prim.IsValid():
                    _api = _PhysxSchema.PhysxContactReportAPI.Apply(_prim)
                    _api.CreateThresholdAttr(0.0)
                    print(f"[gear-env] contact-report API on {_bp}", flush=True)

            # Contact sensor on hand + fingers, filtered to the object and rack so
            # net_forces_w reports the grasp/place contact force (panda_hand alone
            # never touches either surface → was reading ~0 N). For the Robotiq the
            # touching bodies are the finger pads/links (regex segments cannot span
            # '/', and the bare hand reads ~0 N, so sense the four finger bodies).
            _sensor_expr = ("/World/envs/env_.*/Robot/panda_hand/(left|right)_(inner|outer)_finger"
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
            if self.cfg.gripper == "robotiq_2f140":
                self._apply_rubber_pads()
                # NOTE: _apply_grip_gravity_off() is NOT called — the no-grav
                # experiment (probes 16-17) broke the staging both ways: the
                # droop-comp predrive command tilts 0.09 rad without the load
                # (25-35 mm arrival misses), and commanding the reached pose
                # instead removed the ARM's own droop comp too (hand sat low,
                # 4000 N staging slam). Revisit only with a full staging
                # recalibration budget; the gravity-on plant post-symmetric-
                # drives is franka-adjacent already.
            self.scene.clone_environments(copy_from_source=False)

            # Cache joint / body indices
            self._arm_ids = list(range(7))
            self._ee_idx  = -1  # resolved lazily in _reset_idx (data not ready at setup time)

        def _apply_rubber_pads(self) -> None:
            """High-friction 'rubber' physics material on the 2F-140 pad collisions and
            the bottle (the real pads are rubber; pair friction then stays ~2.0
            regardless of the combine mode). Part of the LIPGRIP grasp calibration."""
            try:
                import omni.usd
                from pxr import UsdShade, UsdPhysics
                stage = omni.usd.get_context().get_stage()
                mat = UsdShade.Material.Define(stage, "/World/RubberPadMat")
                UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
                m = UsdPhysics.MaterialAPI(mat.GetPrim())
                m.CreateStaticFrictionAttr(2.0); m.CreateDynamicFrictionAttr(2.0)
                n = 0
                for prim in stage.Traverse():
                    pth = prim.GetPath().pathString
                    if prim.HasAPI(UsdPhysics.CollisionAPI) and (
                            ("Robot" in pth and "inner_finger" in pth) or "/Object" in pth):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
                        n += 1
                print(f"[rubber] bound high-friction material to {n} collision prims")
            except Exception as e:
                print("[rubber] skipped:", e)

        def _apply_grip_gravity_off(self) -> None:
            """Disable gravity on the 2F-140 subtree bodies = model a PERFECT
            tool-gravity compensation (what the franka effectively has — its
            hand gravity is correctly handled by PhysX+OSC). The loop-jointed
            2F-140 breaks PhysX's generalized gravity comp; the scalar ag
            integrator patches the lift but a j6-pitch residual remains and
            the ag<->j6 dance is the ±2-5 mm wobble that wedges every
            insertion at mu 1.0 (probe 15: even perfect P-aim + sustained
            press stalls; v5a/v5b: PPO cannot out-explore it). Task physics
            (contact, friction, break forces, the 30 g gear's weight) are
            untouched — this only removes a sim artifact the controller could
            never compensate."""
            try:
                import omni.usd
                from pxr import UsdPhysics, PhysxSchema
                stage = omni.usd.get_context().get_stage()
                _grip_names = ("robotiq_base_link", "outer_knuckle",
                               "outer_finger", "inner_finger", "inner_knuckle")
                n = 0
                for prim in stage.Traverse():
                    pth = prim.GetPath().pathString
                    if "Robot" in pth and any(g in pth for g in _grip_names) \
                            and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                        px.CreateDisableGravityAttr(True)
                        n += 1
                print(f"[grip-nograv] disabled gravity on {n} gripper bodies",
                      flush=True)
            except Exception as e:
                print("[grip-nograv] skipped:", e)

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
            if self.cfg.budget_mode == "fixed":
                bud = torch.full((n,), self.cfg.budget_fixed_n, device=self.device)
            elif self.cfg.budget_mode == "oracle":
                bud = (fb - self.cfg.oracle_margin_n).clamp(1.0, 120.0)
            else:
                bud = self._obj_budget[cls]
            self._budget_env[ids] = bud
            self._f_cmd[ids]   = bud
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

        def _raw_insertion_vec3(self) -> torch.Tensor:
            """Pairwise Object↔Rack contact force VECTOR (N,3) — grip-free."""
            if getattr(self, "_insert_sensor", None) is None:
                return torch.zeros(self.num_envs, 3, device=self.device)
            data = getattr(self._insert_sensor, "data", None)
            fm = getattr(data, "force_matrix_w", None) if data is not None else None
            if fm is not None and fm.numel() > 0:
                return fm.reshape(self.num_envs, -1, 3).sum(dim=1)
            return torch.zeros(self.num_envs, 3, device=self.device)

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
            if self.cfg.gripper != "robotiq_2f140":
                # GEAR PORT — ARRIVAL-gated stage split for the franka too: the
                # time-based half handed off mid-travel on the gear scene's longer
                # descent leg (probe: EE y 5 cm short of the shaft at setup end).
                _gv_off = self._obj.data.root_pose_w[:, :2] - ee_pos_w[:, :2]
                _dest_xy = torch.stack(
                    [orig[:, 0] + c.rack_x + self._start_off[:, 0],
                     orig[:, 1] + c.rack_y + self._start_off[:, 1]], dim=-1) - _gv_off
                _fk_dxy = (ee_pos_w[:, :2] - _dest_xy).norm(dim=-1)
                stage1 = stage1 | (_fk_dxy > 0.003)  # stay up-&-over until the GEAR is above the shaft
            if self.cfg.gripper == "robotiq_2f140":
                # ARRIVAL-gated descent: drop from the carry altitude only once
                # the EE is over the cell xy — the time-based split descended
                # 7 cm short and drove the carried bottle into the rack wall at
                # 26 N (smoke 19). The impedance-limited carry is ~2 cm/s, so
                # time-based scheduling can't know when the xy leg is done.
                # GEAR PORT — GEAR-AIM (the franka branch's _gv_off, robotiq
                # twin): aim the EE so the GEAR lands over the shaft. The rigid
                # top-down grip hangs the gear ~directly under the EE with a
                # small constant lean — no bottle-pendulum feedback risk. Only
                # once the seat is done (before that the gear is parked/pinned
                # and the offset is garbage; targets are inert then anyway —
                # drive mode / seat freeze — but keep the math clean).
                _rq_gv = torch.zeros(self.num_envs, 2, device=self.device)
                if int(self._rq_seatctr.max()) == 0 and int(self._rq_seatctr.min()) == 0:
                    # IN-HAND gate: gear-aim's target is hand + (shaft - gear),
                    # which for a gear NOT in the grip is a perpetual carrot a
                    # constant offset ahead of the hand — the table-pick probe's
                    # failed grasp sent the arm on a circular runaway to
                    # (-0.58, 0.12). A grip on the Ø35.5 hub stalls the finger
                    # drive well below full-close; closed-on-air reads ~0.784.
                    _held = (self._robot.data.joint_pos[:, self._grip_ids[0]]
                             < 0.74)
                    _rq_gv = torch.where(
                        _held.unsqueeze(-1),
                        self._obj.data.root_pose_w[:, :2] - ee_pos_w[:, :2],
                        _rq_gv)
                _dxy = torch.stack(
                    [orig[:, 0] + c.rack_x + self._start_off[:, 0] + self._rq_dest_dx,
                     orig[:, 1] + c.rack_y + self._start_off[:, 1]], dim=-1) - _rq_gv
                if self.cfg.rq_table_pick:
                    # XY standing-error integrator (table pick): the OSC holds
                    # the arm at appr - F_res/k (probe-11 rule), and with the
                    # pin gone the REAL gear inherits that ~12 mm offset — the
                    # 8 mm arrival gate never fired (probe 6 hover). Integrate
                    # the gear-vs-shaft error into the carrot exactly like the
                    # z integrator; only while carrying (gripped, post-lift)
                    # and near the destination (no traverse wind-up).
                    _sxy_t = torch.stack(
                        [orig[:, 0] + c.rack_x + self._start_off[:, 0],
                         orig[:, 1] + c.rack_y + self._start_off[:, 1]], dim=-1)
                    _gerr = self._obj.data.root_pose_w[:, :2] - _sxy_t
                    _held_t = (self._robot.data.joint_pos[:, self._grip_ids[0]]
                               < 0.74)
                    if (not self._rq_drive_mode) and self._rq_lift_ok:
                        _int_on = (_held_t & (self._setup_ctr > 0)
                                   & (self._rec_steps == 0)
                                   & (_gerr.norm(dim=-1) < 0.05))
                        self._rq_xyint = torch.where(
                            _int_on.unsqueeze(-1),
                            (self._rq_xyint + 0.008 * _gerr).clamp(-0.04, 0.04),
                            self._rq_xyint)
                    # NOTE: the bias applies to the OSC carrot `appr` (the
                    # setup target), NOT to _dxy — _dxy feeds the stage/arrival
                    # gating and biasing it would corrupt the gate semantics
                if self._rq_centered:
                    # POST-RECOVERY: base-aim — steer the EE so the bottle
                    # base lands on the cell center. Replaces the static bias
                    # counters (+x 0.055 / -y 0.075): the arm's persistent
                    # P-push differs per maneuver (rotate_align vs retract), so
                    # open-loop offsets calibrated on one flow re-hit the rim on
                    # the other — and each rim press ratchets the neck through
                    # the pads until the bottle drops (render loop 2, 6/6).
                    # FROZEN at the recovery falling edge, NOT live (loop 12:
                    # live feedback pumps the bottle pendulum — see rec_end).
                    # An alpha=0.01 low-pass refresh was tried (take 90) to
                    # track the ori_k-110 wrist-twist drift that displaces
                    # the hanging bottle ~5 cm (take 91's attempt-1 divider
                    # grind): it chased the LEAN and delivered the insertion
                    # tilted — the bottle seated +4 cm off-center against the
                    # cell wall and TOPPLED on release. The frozen aim seats
                    # clean and vertical (take 91 attempt 2); a twist-induced
                    # miss just costs one extra recovery cycle, which k_max
                    # absorbs and the jam detector handles honestly.
                    _dxy = self._rq_aim
                # SWING-SETTLE GATE (post-recovery only): the retract maneuver
                # leaves the held bottle swinging ±8 cm; starting the re-descent
                # (or even chasing the aim xy) while it swings PUMPS the pendulum
                # under render tracking until the bottle flings out of the pinch
                # (takes 98/99 — smoke sag damps the same swing, so smokes pass).
                # Hold the EE still (zero commanded xy motion = zero energy in)
                # until |v_xy| stays calm for 30 substeps, then descend normally.
                _swv = self._obj.data.root_vel_w[:, :2].norm(dim=-1)
                self._rq_calmctr = torch.where(
                    _swv < 0.08, self._rq_calmctr + 1,
                    torch.zeros_like(self._rq_calmctr))
                # release the settle request when the bottle has been calm for
                # 30 consecutive substeps, OR unconditionally after 600 substeps
                # (livelock guard: a bottle that never reads calm under render
                # physics must not hold the attempt to its 2375-step timeout)
                self._rq_settlewait = torch.where(
                    self._rq_need_settle, self._rq_settlewait + 1,
                    torch.zeros_like(self._rq_settlewait))
                self._rq_need_settle = (self._rq_need_settle
                                        & (self._rq_calmctr < 30)
                                        & (self._rq_settlewait < 600))
                _settling = (torch.full_like(self._rq_desc, self._rq_centered)
                             & self._rq_need_settle & (self._setup_ctr > 0))
                _dxy = torch.where(_settling.unsqueeze(-1),
                                   self._rq_settle_pos[:, :2], _dxy)
                self._rq_settling = _settling
                _dist = (ee_pos_w[:, :2] - _dxy).norm(dim=-1)
                # PERMANENT latch: engage the descent at 2 cm and never release —
                # a release bound (smoke 22: 6 cm) re-commanded the +20 cm climb
                # mid-descent when xy drifted, and the up-target is what triggers
                # every float-away. If xy drifts during the descent, the P pulls
                # it back at the LOW altitude (over the open cell — benign).
                _newly_desc = (_dist < 0.02) & ~_settling & ~self._rq_desc
                if bool(_newly_desc.any()) and self._joint_centers is not None:
                    # GEAR PORT: re-anchor the null-space posture to the CURRENT
                    # branch when the descent latches. The handoff snapshot is
                    # 0.59 rad away after the traverse (stage probe 3 jc_dist)
                    # and the gain-15 posture pull leaks into task space as a
                    # constant push — 9 mm xy offset + slow tilt creep at the
                    # entrance hold (the take-96 mechanism, construction-gain
                    # edition). Zero initial pull, same as the rec_end fix.
                    self._joint_centers = torch.where(
                        _newly_desc.unsqueeze(-1),
                        self._robot.data.joint_pos[:, self._arm_ids],
                        self._joint_centers)
                self._rq_desc = self._rq_desc | ((_dist < 0.02) & ~_settling)
                stage1 = (stage1 | (_dist > 0.03)) & (~self._rq_desc)
                # CONTACT-TERMINATED descent: stop at first rim touch (~3 N) —
                # the franka's trained hand-off state is bottle-PRESSED-ON-RIM.
                # A fixed altitude either leaves free space (policy flees, smoke
                # 27) or lets the bottle slide into the hole under SCRIPT (smoke
                # 28's disallowed "success"). On touch: freeze the altitude and
                # fast-forward the setup so the LEARNED policy takes over.
                # INSTANTANEOUS force (not _cf_filt): the filter lag let the
                # crawling descent keep digging for dozens of substeps after
                # true contact — at kpos 600 that's a 35-39 N impact spike on a
                # 21 N-break glass (smoke 47). The raw signal trips the freeze
                # within a couple of substeps of first touch.
                # gate threshold: 1.5 N on the first (wedge) descent — freeze at
                # the faintest rim kiss. 3.0 N once centered: the funnel ride
                # brushes the wall at 1-2 N while self-centering, and freezing on
                # a brush strands the bottle mid-mouth (the policy then presses
                # a false 6-7 N jam — smoke 50 attempt 1 — spawning extra
                # recovery cycles that ratchet the bottle out of the pinch).
                # PLUS a sustained-press latch when centered: render loop 4 sat
                # at a constant 2.85 N — under the 3.0 N gate, under the jam
                # threshold — for 2400 substeps and timed out three takes in a
                # row. Amplitude alone can't split brush from stall at render
                # impedance; DURATION can — a funnel brush is transient while
                # the bottle keeps descending, a stall holds contact steadily.
                # 60 consecutive substeps > 1.5 N ==> latched contact.
                _thr = 3.0 if self._rq_centered else 1.5
                # NEAR-FUNNEL guard on both latches: contact only counts as
                # "rim" if the bottle base is within 12 cm of the cell center
                # (covers the deliberate 8 cm wedge; excludes rack-EDGE
                # grazes — render loop 9 latched rimz on a 2 N graze 19 cm
                # off and the policy took over stranded on the outer wall).
                # (A/B'd out in smoke 77: not the attempt-1 regression.)
                _bnear = (self._obj.data.root_pose_w[:, :2]
                          - torch.stack([orig[:, 0] + c.rack_x,
                                         orig[:, 1] + c.rack_y], dim=-1)
                          ).norm(dim=-1) < 0.12
                _press = (self._rq_desc & (self._rq_rimz < 0) & _bnear
                          & (self._cf_insert > 1.5))
                self._rq_pressctr = torch.where(
                    _press, self._rq_pressctr + 1,
                    torch.zeros_like(self._rq_pressctr))
                _seat_done = (self._rq_seatctr == 0)
                _hit = self._rq_desc & (self._rq_rimz < 0) & _bnear & _seat_done & (
                    (self._cf_insert > _thr) | (self._rq_pressctr >= 60))
                # freeze 5 mm ABOVE the touch point: the descent's momentum
                # carries the press to ~12 N, which exceeds the pinch's AXIAL
                # SLIP limit — the bottle slides up through the pads and the
                # contact unloads to zero (smoke 33). Backing off settles the
                # press at ~3-5 N: sustained, in the policy's trained band.
                # (Freezing 1.2 cm BELOW pushed the base off the divider into
                # the hole — smoke 31; AT the point overpressed — smoke 33.)
                self._rq_rimz = torch.where(_hit, ee_pos_w[:, 2] + 0.005, self._rq_rimz)
                # hand off FAST (12 substeps, was 60): the RQ_TRACE2 render
                # trace showed the whole over-press builds INSIDE this
                # countdown — cf 3.9 -> 29 N while the pads slid 2 cm along
                # the bottle (squeeze-extrusion feedback: deeper slip ->
                # thicker neck in the pinch -> harder squeeze -> more press).
                # The lam-clamped setup impedance can only pull up 4.8 N and
                # cannot stop it; the LEARNED policy can and does — at
                # setup==0 it unloaded 27.5 -> 6.5 N in ~10 substeps. So give
                # it control while the press is still ~10 N.
                self._setup_ctr = torch.where(
                    _hit, torch.minimum(self._setup_ctr,
                                        torch.full_like(self._setup_ctr, 12)),
                    self._setup_ctr)
                # GEAR PORT — ARRIVAL fast-forward (free-space hand-off): the
                # franka gear hand-off is FREE SPACE at the entrance (gear over
                # the shaft at ~0.43-0.46), not the bottle's pressed-on-rim
                # state, so the contact-terminated logic above never fires on a
                # clean approach and the setup would idle out its full window.
                # Hand off to the policy once the GEAR is over the shaft at
                # entrance height (franka gate: xy < 3 mm, z < 0.465).
                _g_xy = (self._obj.data.root_pose_w[:, :2]
                         - torch.stack([orig[:, 0] + c.rack_x + self._start_off[:, 0],
                                        orig[:, 1] + c.rack_y + self._start_off[:, 1]],
                                       dim=-1)).norm(dim=-1)
                _g_z = self._obj.data.root_pose_w[:, 2] - orig[:, 2]
                # table pick: 0.465 was slack tuned for the pinned flow (the pin
                # delivered the gear at 0.430-0.446 regardless); the table carry
                # descends FROM ABOVE, so the slack gate fires at ~0.46 and the
                # policy inherits an OOD free hover above its trained band and
                # drifts (probe 8). Hand off only inside the band.
                _g_ztop = 0.448 if self.cfg.rq_table_pick else 0.465
                # xy 4.5 mm (not 3) in table mode: the unpinned hover's best is
                # ~3.2 mm (probe 10 sat 0.5 mm outside the gate until the setup
                # window expired and the policy inherited a 16 mm hand-off);
                # force-guided search captures from 3-4 mm.
                _g_xytol = 0.0045 if self.cfg.rq_table_pick else 0.003
                _g_arr = (self._rq_desc & (self._rq_rimz < 0) & _seat_done
                          & (_g_xy < _g_xytol) & (_g_z < _g_ztop))
                if self.cfg.rq_table_pick:
                    # hand off SETTLED and NEAR-UPRIGHT: the carry arrives with
                    # residual swing and in-grip lean; the pinned flow delivered
                    # ~0 deg, and at 3.5 deg the policy honestly refuses the
                    # geometrically impossible insertion and retreats (probe 9)
                    _g_q = self._obj.data.root_pose_w[:, 3:7]
                    _g_upr = (1.0 - 2.0 * (_g_q[:, 1] ** 2 + _g_q[:, 2] ** 2)
                              ) > 0.99863   # cos(3 deg)
                    _g_still = self._obj.data.root_vel_w[:, :3].norm(dim=-1) < 0.03
                    _g_arr = _g_arr & _g_upr & _g_still
                self._setup_ctr = torch.where(
                    _g_arr, torch.minimum(self._setup_ctr,
                                          torch.full_like(self._setup_ctr, 12)),
                    self._setup_ctr)
                import os as _os6h
                self._holdct = getattr(self, "_holdct", 0) + 1
                if (_os6h.environ.get("RQ_TRACE") == "1" and bool(self._rq_desc[0])
                        and self._holdct % 200 == 0):
                    # GEAR PORT hold diagnosis: who moves the arm during the
                    # free-space entrance hold (run-2 creep: handdn .998->.959)?
                    _lq = self._robot.data.body_quat_w[0, self._ee_idx]
                    _oerr = float(2.0 * torch.acos(
                        (_lq * self._ee_quat_des[0]).sum().abs().clamp(max=1.0)))
                    _jc = (self._robot.data.joint_pos[0, :7]
                           - self._joint_centers[0]).norm() \
                        if self._joint_centers is not None else torch.tensor(0.0)
                    print(f"      [rq-hold] gxy={float(_g_xy[0])*1000:.1f}mm "
                          f"gz={float(_g_z[0]):.4f} "
                          f"eexy=({float(ee_pos_w[0,0]-orig[0,0]):.4f},"
                          f"{float(ee_pos_w[0,1]-orig[0,1]):.4f}) "
                          f"eez={float(ee_pos_w[0,2]-orig[0,2]):.4f} "
                          f"oerr={_oerr:.4f} ag={float(self._rq_ag[0]):+.3f} "
                          f"jc_dist={float(_jc):.3f}", flush=True)
                if bool(_hit[0]):
                    import os as _os6
                    if _os6.environ.get("RQ_TRACE") == "1":
                        print(f"      [rq-rim] contact at ee_z="
                              f"{float(ee_pos_w[0, 2] - orig[0, 2]):.3f} "
                              f"cf={float(self._cf_filt[0]):.1f} -> policy in 60",
                              flush=True)
                import os as _os6b
                self._t2ct = getattr(self, "_t2ct", 0) + 1
                if (_os6b.environ.get("RQ_TRACE2") == "1"
                        and bool(self._rq_desc[0])
                        and (float(ee_pos_w[0, 2] - orig[0, 2]) < 0.72
                             or self._rq_centered)
                        and (self._t2ct % 10 == 0
                             or float(self._cf_insert[0]) > 0.5)):
                    # DEBUG per-substep contact trace (render diagnosis only)
                    _bp = self._obj.data.root_pose_w[0, :3]
                    _gp = float(self._robot.data.joint_pos[0, self._grip_ids[0]])
                    print(f"      [rq-c] ee=({float(ee_pos_w[0,0]-orig[0,0]):.3f},"
                          f"{float(ee_pos_w[0,1]-orig[0,1]):.3f},"
                          f"{float(ee_pos_w[0,2]-orig[0,2]):.4f})"
                          f" cf={float(self._cf_insert[0]):6.2f}"
                          f" b=({float(_bp[0]-orig[0,0]):.3f},"
                          f"{float(_bp[1]-orig[0,1]):.3f},"
                          f"{float(_bp[2]-orig[0,2]):.4f})"
                          f" grip={_gp:.4f}"
                          f" rimz={float(self._rq_rimz[0]):.3f}"
                          f" setup={int(self._setup_ctr[0])}"
                          f" ctr={int(self._rq_centered)}"
                          f" stl={int(self._rq_need_settle[0])}"
                          f"/{int(self._rq_calmctr[0])}", flush=True)
            appr = ee_pos_w.clone()
            appr[:, 0] = orig[:, 0] + c.rack_x + self._start_off[:, 0] + self._rq_dest_dx
            appr[:, 1] = orig[:, 1] + c.rack_y + self._start_off[:, 1]
            if self.cfg.gripper == "robotiq_2f140":
                appr[:, :2] = appr[:, :2] - _rq_gv   # GEAR-AIM (see _dxy note)
                if self.cfg.rq_table_pick:
                    # standing-error integrator bias (see the _rq_xyint note)
                    appr[:, :2] = appr[:, :2] - self._rq_xyint
            if self.cfg.gripper != "robotiq_2f140":
                # GEAR-AIM: steer the EE so the GEAR (not the hand) is over the
                # shaft — see _gv_off note above.
                appr[:, :2] = _dest_xy
                # Setup descent force budget: the FIXED low appr target saturates
                # the rate limiter (error > lam) so the impedance pulls its full
                # lam x kpos = 15 N at kpos 600 — a rolling 1 cm crawl target
                # capped it at 6 N and the descent stalled (probes guided2/3).
                # Only intervention: FREEZE the pull when insertion contact
                # reaches 60% of the budget (the post-handoff budget freeze
                # does not run during setup).
                _hold = ((~stage1) & (self._setup_ctr > 0)
                         & (self._cf_insert >= 0.6 * self._budget_env))
                appr[:, 2] = torch.where(_hold, ee_pos_w[:, 2], appr[:, 2])
            if self.cfg.gripper == "robotiq_2f140" and self._rq_centered:
                # frozen base-aim carrot (see the _dxy / rec_end notes)
                appr[:, :2] = self._rq_aim
            # _tcp_dz shifts the HAND heights for longer grippers (robotiq) so the
            # BOTTLE traverses/hands-off at the same altitudes as with the panda hand.
            _tz = c.transport_z + (self._rq_carry_dz if self.cfg.gripper == "robotiq_2f140" else 0.0)
            appr[:, 2] = orig[:, 2] + torch.where(stage1, torch.full_like(appr[:, 2], _tz),
                                                  torch.full_like(appr[:, 2], c.forge_approach_z)) \
                         + self._tcp_dz
            if self.cfg.gripper == "robotiq_2f140":
                if self._rq_centered:
                    # centered (post-recovery) descent: DEEPER than the entrance
                    # — the funnel ride is contact-free in the middle and the
                    # burst must carry the bottle to the SEAT (smoke 45: stopped
                    # at the entrance altitude with the bottle mid-mouth, and
                    # the free-space policy lifted it back out). The 3 N contact
                    # gate still halts this instantly on any resistance.
                    # -0.10, not -0.06: the seat is ~6 cm of bottle travel, so a
                    # -0.06 carrot has ZERO slack — under the render pipeline's
                    # extra per-capture physics stepping the impedance realizes
                    # a couple cm high and the bottle dangles at 0.25 N above
                    # the seat, too light for the gate OR the jam detector:
                    # every post-recovery attempt timed out (render loop 3).
                    appr[:, 2] = appr[:, 2] - 0.10
                # rim-contact altitude freeze (set below on first ~3 N touch)
                appr[:, 2] = torch.where(self._rq_rimz > 0,
                                         self._rq_rimz, appr[:, 2])
            pol = ee_pos_w + self._actions[:, :3] * c.forge_act_range  # learned EE delta (gentle)
            if self.cfg.gripper == "robotiq_2f140":
                # ANCHORED policy target (plant fix, probe_rq_insert_plant):
                # the ee-rebaselined delta above means zero action = zero
                # restoring force, and the 2F-140's ~2.3 N unmodeled wrist
                # disturbance walks the free arm ~1 mm/step from the hand-off
                # (probe case B: 3 mm -> 240 mm, no contact; the fence never
                # stopped it). Integrate the policy's delta into a PERSISTENT
                # anchor instead: zero action now HOLDS position with real
                # static stiffness, and every rq checkpoint's press-at-20 N
                # habit (the only self-made anchor available before) becomes
                # unnecessary. Anchor re-arms whenever the env re-enters setup
                # (episode reset or recovery re-approach).
                setup_now = self._setup_ctr > 0
                self._pol_anchor_set &= ~setup_now
                _newly = (~setup_now) & (~self._pol_anchor_set)
                if bool(_newly.any()):
                    # Anchor to the setup's COMMANDED target (appr), NOT the
                    # realized EE (probes 9-11 + pc-trace): the servo holds
                    # the arm at appr - F_res/k — the standing tracking error
                    # is load-bearing (it carries the model-residual force,
                    # measured: no payload moment, no saturation, ag 0, yet
                    # anchoring to the EE walked every env identically 35 mm
                    # to the residual's new equilibrium). Command continuity
                    # across the boundary = nothing moves at hand-off. Use the
                    # last realized OSC command, NOT the mid-function appr
                    # (that one carries the -0.10 descent carrot, probe 12).
                    _lct = getattr(self, "_last_cmd_target", None)
                    src = (_lct if _lct is not None else ee_pos_w).clone()
                    # z: the last setup command carries the -0.10 descent
                    # carrot (saturated, 9-14 mm below the settled arm) — a
                    # step input that rings the wobble exactly as the policy
                    # starts its descent (shim-fix triage: contact arrives
                    # 4-8 mm off, outside the 1.5 mm funnel -> wedge-shed at
                    # the +y geometric floor). Keep xy command continuity,
                    # clamp the z-init to just below the current EE.
                    src[:, 2] = torch.maximum(src[:, 2], ee_pos_w[:, 2] - 0.003)
                    self._pol_anchor[_newly] = src[_newly]
                    self._pol_anchor_set |= _newly
                    import os as _osA
                    if _osA.environ.get("RQ_TRACE") == "1" and bool(_newly[0]):
                        print(f"      [rq-anchor] init=("
                              f"{float(src[0,0]-orig[0,0]):.3f},"
                              f"{float(src[0,1]-orig[0,1]):.3f},"
                              f"{float(src[0,2]-orig[0,2]):.4f}) "
                              f"ee=({float(ee_pos_w[0,0]-orig[0,0]):.3f},"
                              f"{float(ee_pos_w[0,1]-orig[0,1]):.3f},"
                              f"{float(ee_pos_w[0,2]-orig[0,2]):.4f})",
                              flush=True)
                    # NOTE (probe 11): do NOT re-anchor the ori command here.
                    # The standing ori error at hand-off is LOAD-BEARING — it
                    # is how the wrist carries the un-modeled gravity moment
                    # (equilibrium error = M/k). Re-anchoring command=physical
                    # left the moment unopposed and the wrist fell to a new
                    # equilibrium anyway (case B walk unchanged). Continuity
                    # is the fix instead: the learned phase keeps the SETUP's
                    # exact ori regime (600, spawn-quat command) so nothing
                    # shifts at the boundary.
                _live_m = ((~setup_now) & self._pol_anchor_set).unsqueeze(-1).float()
                self._pol_anchor = self._pol_anchor \
                    + self._actions[:, :3] * c.forge_act_range * _live_m
                # anti-windup on Z ONLY: the anchor may not lead the realized
                # EE by >2 cm vertically (at training std 0.5 a deep z windup
                # was a BREAK MACHINE — the wound anchor slammed the gear into
                # the plate whenever it slipped off the shaft, v5 run 1). XY
                # gets NO windup tracking: re-centering the anchor on the EE
                # let contact reactions DRAG the command 66 mm sideways
                # (descend-trace 6 ratchet); lateral bounds belong to the
                # fence, which already clamps the target into the chimney.
                self._pol_anchor[:, 2] = ee_pos_w[:, 2] + (
                    self._pol_anchor[:, 2] - ee_pos_w[:, 2]).clamp(-0.02, 0.02)
                # CONTACT LEAK (v5 postmortem): the anchor INTEGRATES
                # exploration noise into sustained deep presses — the franka's
                # ee-rebaselined target forgets a noise-press next step, ours
                # holds it, and at training std 0.5 the abs force penalty
                # bled until disengagement won. Above 0.8x the force budget
                # the anchor z decays toward the EE (noise presses become
                # self-limiting); steel's legit 100 N press keeps its full
                # budget headroom.
                _lk = ((self._cf_insert > 0.8 * self._budget_env).float() * 0.15)
                self._pol_anchor[:, 2] = self._pol_anchor[:, 2] + _lk * (
                    ee_pos_w[:, 2] - self._pol_anchor[:, 2])
                pol = torch.where(self._pol_anchor_set.unsqueeze(-1), self._pol_anchor, pol)
            else:
                pol = torch.cat([pol[:, :2] + self._fk_aim, pol[:, 2:3]], dim=-1)
            if self.cfg.gripper == "robotiq_2f140" and self.cfg.rq_table_pick:
                # SETTLE-HOVER HOLD: once the carry first reaches the arrival
                # ball (xy + z; the gear may still lean), FREEZE the setup
                # target — a fixed spring resists the slow wrist-twist +y
                # drift that outran gear-aim + the xy integrator (probes
                # 10/11: y crept 15 mm while the tilt settled and the gate
                # never fired). Fixed-snapshot precedent: _rq_seat_hold
                # (take-97 rule — live-ee holds have zero restoring force,
                # a FROZEN snapshot has real stiffness). Released at setup
                # end; re-latches fresh on a recovery re-approach.
                if not hasattr(self, "_rq_tp_hold_on"):
                    self._rq_tp_hold_on = torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device)
                    self._rq_tp_hold = torch.zeros(
                        self.num_envs, 3, device=self.device)
                _tp_gxy = (self._obj.data.root_pose_w[:, :2] - torch.stack(
                    [orig[:, 0] + c.rack_x + self._start_off[:, 0],
                     orig[:, 1] + c.rack_y + self._start_off[:, 1]],
                    dim=-1)).norm(dim=-1)
                _tp_gz = self._obj.data.root_pose_w[:, 2] - orig[:, 2]
                if (not self._rq_drive_mode) and self._rq_lift_ok:
                    _tp_enter = ((self._setup_ctr > 0) & (~self._rq_tp_hold_on)
                                 & (self._rq_seatctr == 0) & (self._rq_rimz < 0)
                                 & (_tp_gxy < 0.0045) & (_tp_gz < 0.448))
                    if bool(_tp_enter.any()):
                        self._rq_tp_hold[_tp_enter] = ee_pos_w[_tp_enter]
                        self._rq_tp_hold_on |= _tp_enter
                self._rq_tp_hold_on &= (self._setup_ctr > 0)
                appr = torch.where(self._rq_tp_hold_on.unsqueeze(-1),
                                   self._rq_tp_hold, appr)
            raw = torch.where(in_setup, appr, pol)
            # Fast traverse during setup, SLOW (low-impulse) approach during the learned
            # insertion so the bottle never hits the cell with breaking momentum.
            # 0.025 (not 0.015): the rate-limited target caps the P restoring
            # force at lam*kpos per axis — 6 N couldn't recover a floating arm
            # once the lift feedforward overshot (smoke 20); 10 N/axis can.
            _setup_lam = 0.025 if self.cfg.gripper == "robotiq_2f140" else c.lam
            # GEAR PORT plant fix (probe_rq_insert_plant case A): the learned-
            # phase rate limit caps the P restoring force at lam*kpos per axis.
            # forge_lam 0.004 * 600 = 2.4 N/axis — BELOW the 2F-140's ~2.3 N
            # hanging-wrist disturbance, so a fully saturated corrective policy
            # recovered 0.13 mm/step from a 180 mm drift (i.e. the plant cannot
            # reject its own disturbance, no policy can learn on it). 0.012
            # gives 7.2 N/axis (~3x the disturbance); insertion gentleness is
            # still bounded by the budget z-freeze + 1.5x hard ceiling.
            # Probe rerun: 0.012 still drifted (0.15-0.5 mm/step, tilt-coupled)
            # while the SETUP servo (same kpos 600 / ori_k 1200, lam 0.025,
            # held absolute target) tracks 0.8 mm statically with zero tilt
            # growth — so the learned phase now runs the setup's exact
            # authority. Gentleness is the policy's job (force-conditioned
            # descent), enforced by the budget z-freeze + break penalty.
            _ins_lam = 0.025 if self.cfg.gripper == "robotiq_2f140" else c.forge_lam
            lam = torch.where(in_setup, torch.full_like(appr[:, :1], _setup_lam),
                              torch.full_like(appr[:, :1], _ins_lam))
            if self.cfg.gripper == "robotiq_2f140":
                # REGIME-SPLIT per-axis authority (probes 3-4): the two regimes
                # need OPPOSITE plants and get their own —
                #  * FREE SPACE: xy 0.025 (hold the aim against the ~2.3 N
                #    wrist disturbance), z 0.008; below the entrance altitude
                #    z crawls at 0.003 so first contact is quasi-static
                #    (probe 4: a 0.008 approach still spiked 118 N impulse and
                #    broke abs gears before the z-freeze could react — and the
                #    slam TILTS THE GEAR IN THE PADS permanently, ~12 deg,
                #    after which seating is geometrically impossible).
                #  * CONTACT (insertion-force present): the FRANKA insertion
                #    plant — 0.004 all axes (2.4 N/axis, the funnel-proven
                #    gentle search authority). Hard lateral shoving while
                #    wedged is what rotates the gear in the 24 N pinch; the
                #    shaft itself carries the gear now, so the free-space
                #    disturbance concern is moot.
                lam = lam.expand(-1, 3).clone()
                _soft = getattr(self, "_ok_soft_ctr", None)
                _gz_l = self._obj.data.root_pose_w[:, 2] - self.scene.env_origins[:, 2]
                # insertion regime = recent contact ONLY (see the ori_k gate
                # note: an altitude term here nerfed the final free-space
                # approach and caused the probe-8 tip-edge shed)
                _inc = (_soft > 0) if _soft is not None else torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device)
                # in-contact xy 0.015 (~9 N/axis), NOT the bottle's 0.004:
                # seat discovery is a PRESS-AND-SLIDE search — dragging a
                # ~10 N press across mu 1.0 needs 10-15 N lateral, and at
                # 2.4 N the contact regime FORBADE the only motion that finds
                # the funnel (v4/v5/v6 all starved of successes). The
                # grinding-eject that 0.004 guarded against was the old tilty
                # plant; pad grip friction (~38 N) holds the gear through a
                # 9 N drag.
                _lam_xy = torch.where(_inc, torch.full_like(_gz_l, 0.015),
                                      torch.full_like(_gz_l, 0.025))
                _lam_z = torch.where(_inc, torch.full_like(_gz_l, 0.004),
                                     torch.where(_gz_l < 0.435,
                                                 torch.full_like(_gz_l, 0.003),
                                                 torch.full_like(_gz_l, 0.008)))
                _su = in_setup.squeeze(-1)
                lam[:, 0] = torch.where(_su, lam[:, 0], _lam_xy)
                lam[:, 1] = torch.where(_su, lam[:, 1], _lam_xy)
                lam[:, 2] = torch.where(_su, lam[:, 2], _lam_z)
            # GEAR PORT: NO robotiq descent slow-down — the gear descent ends in
            # FREE SPACE at the entrance (hand-off gate xy<3mm, z<0.465);
            # contact gentleness is the LEARNED policy's job after hand-off
            # (forge_lam 0.004), exactly like the franka setup (c.lam 0.025
            # at kpos 600 = 15 N/axis). The bottle's 0.008 crawl halved the
            # restoring authority to 4.8 N/axis and the free-space hold
            # destabilized (stage probe 1: +y yield to 40 cm under the
            # ag-rail distortion).
            # After the policy releases, the bottle is placed — the hand no longer needs the
            # slow gentle-insertion motion cap, so retract at a MODERATE speed (the full setup
            # lam yanks the arm and spikes joint forces; 1.5 cm/step pulls clear smoothly).
            if self.cfg.forge_release_mode:
                lam = torch.where(self._released.unsqueeze(-1),
                                  torch.full_like(lam, 0.015), lam)
            delta = torch.maximum(torch.minimum(raw - ee_pos_w, lam), -lam)
            # GEAR PORT: the bottle's CONTACT SLOW ZONE (down-steps <= 0.003 in
            # the last 2 cm above wedge contact) is removed — the gear descent
            # terminates in free space above the shaft tip; there is no
            # scripted touchdown to soften. (Bottle history: render loops 4-6
            # impact spikes; smokes 59/60 for the asymmetric-clamp shape.)
            target = ee_pos_w + delta
            if self.cfg.gripper == "robotiq_2f140":
                # SWING-SETTLE HOLD: while settling, the target IS the fixed
                # snapshot — all three axes. The _dxy override alone still let
                # the z descent crawl (desc stays latched through recovery) and
                # dragged the swinging bottle down toward the rim (take 97).
                _st = getattr(self, "_rq_settling", None)
                if _st is not None:
                    target = torch.where(_st.unsqueeze(-1),
                                         self._rq_settle_pos, target)
                import os as _os6c
                if (_os6c.environ.get("RQ_TRACE2") == "1"
                        and self._rq_centered
                        and getattr(self, "_t2ct", 0) % 10 == 0):
                    # DEBUG: the ACTUAL commanded carrot (render diagnosis —
                    # take 96's EE spiraled +y away from the frozen aim)
                    print(f"      [rq-t] tgt=({float(target[0,0]-orig[0,0]):.3f},"
                          f"{float(target[0,1]-orig[0,1]):.3f},"
                          f"{float(target[0,2]-orig[0,2]):.4f})"
                          f" raw=({float(raw[0,0]-orig[0,0]):.3f},"
                          f"{float(raw[0,1]-orig[0,1]):.3f},"
                          f"{float(raw[0,2]-orig[0,2]):.4f})"
                          f" aim=({float(self._rq_aim[0,0]-orig[0,0]):.3f},"
                          f"{float(self._rq_aim[0,1]-orig[0,1]):.3f})"
                          f" desc={int(self._rq_desc[0])}"
                          f" st1={int(stage1[0])}"
                          f" stl={int(_st[0]) if _st is not None else -1}"
                          f" oerr={float(2.0 * torch.acos((self._robot.data.body_quat_w[0, self._ee_idx] * self._ee_quat_des[0]).sum().abs().clamp(max=1.0))):.3f}",
                          flush=True)
            # ── FORGE force authority (safety, NOT the policy): when axial contact
            # reaches the per-object budget, FREEZE the descent (don't push the EE
            # lower) — but leave xy free so the policy can still slide/align, and
            # don't retreat (so resting on the cell floor = seated, not undone).
            # Compliant stiffness (forge_pos_k) keeps lateral rams survivable. This
            # bounds the force to ~F_max without preventing the gentle insertion.
            budget = self._budget_env
            over = (self._cf_insert > budget) & (self._setup_ctr == 0)   # insertion force only
            if self.cfg.forge_release_mode:
                over = over & (~self._released)   # after release, never freeze z — let the hand lift away
            frozen_z = torch.maximum(target[:, 2], ee_pos_w[:, 2])
            target[:, 2] = torch.where(over, frozen_z, target[:, 2])
            # HARD fragile ceiling (render loops 4/5): the freeze above never
            # RETREATS — it re-baselines at the current ee each substep, so the
            # render pipeline's capture-interleaved physics digs a few mm per
            # capture and the press ratchets 8.8 -> 17.7 N, past the pinch's
            # axial slip limit: the bottle slid out of the pads at the attempt-0
            # jam and every later attempt ran with an empty gripper over a
            # resting bottle (constant cf = weight 2.85 N -> timeout). Above
            # 1.5x budget, retreat at least 2 mm/substep. torch.maximum, NOT a
            # replacement (render loop 6/7 lesson): overwriting the target with
            # ee+0.002 CUT the up-pull to 1.2 N whenever the normal path
            # already commanded higher (the lam-clamped rimz freeze = 4.8 N up)
            # and the press ran away to 29 N. Armed in ALL phases. Jam
            # detection is unaffected: thresh = max(6.0, 0.18*F_max) N.
            _hard = self._cf_insert > 1.5 * budget
            if self.cfg.forge_release_mode:
                _hard = _hard & (~self._released)
            target[:, 2] = torch.where(
                _hard, torch.maximum(target[:, 2], ee_pos_w[:, 2] + 0.008),
                target[:, 2])
            if self.cfg.gripper == "robotiq_2f140":
                # WORKSPACE FENCE (LEARNED phase): clamp the policy's target to a
                # box around the cell. In free space the policy's action mean
                # saturates away from the task (trained in-contact only) and one
                # contact loss used to end the episode in a cross-room swing
                # (smokes 26-38). Inside the box the policy has full authority;
                # the fence only stops runaways so every attempt stays
                # recoverable. Not motion scripting — a work-envelope limit.
                # GEAR PORT: fence centered ON the shaft xy — the top-down EE
                # hangs the gear directly below it. (The bottle's side grip put
                # the EE one tcp-length (-0.214) behind the bottle, hence the
                # old -_grasp_tcp_d term, which here would clamp the policy
                # 21 cm short of the shaft.)
                _fc = torch.stack([orig[:, 0] + c.rack_x + self._start_off[:, 0],
                                   orig[:, 1] + c.rack_y + self._start_off[:, 1],
                                   orig[:, 2] + c.forge_approach_z + self._tcp_dz], dim=-1)
                # (fence stays NOMINAL-centered. An aim-following chimney —
                # tried after render loop 11 parked at the fence's y edge —
                # was present in every configuration that lost the direct
                # attempt-1 seat (smokes 65-75): with a live aim the clamp
                # box sways with the bottle pendulum and the policy phase
                # hovers sub-gate. The direct-seat configs (smokes 61-63)
                # all ran the nominal fence; loop 11's edge-park was the
                # 0.015 maneuver slow-down's misalignment, not the fence.)
                # lateral half-width: 0.10 pre-jam (room for the seeded wedge at
                # +0.08), 0.04 once the recovery has centered the approach — a
                # chimney over the goal cell, so every contact burst the policy
                # gets lands within the funnel's capture radius (at ±0.10 it
                # parked on the +y corner over a neighbor divider, smoke 41).
                _lat = 0.04 if self._rq_centered else 0.10
                _flo = _fc + torch.tensor([-_lat, -_lat, -0.25], device=self.device)
                # INSERTION-MODE ceiling: once rim contact is established, the
                # policy may not retreat more than 1 cm above it — its free-space
                # habit is to lift out of engagement and wander (smoke 40, att 1).
                # Lateral + downward authority (the actual insertion work) stays
                # full; the recovery maneuver overrides the fence when a real
                # retreat is commanded (its target is applied after this clamp).
                _zhi = torch.where(self._rq_rimz > 0, self._rq_rimz + 0.010,
                                   _fc[:, 2] + 0.05)
                _fhi = torch.cat([_fc[:, :2] + _lat, _zhi.unsqueeze(-1)], dim=-1)
                _live = (self._setup_ctr == 0).unsqueeze(-1)
                target = torch.where(_live, torch.max(torch.min(target, _fhi), _flo),
                                     target)

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
                    clear[:, 0] = orig[:, 0] + c.rack_x + self._start_off[:, 0] + rec[:, 0] \
                                  + self._rq_dest_dx
                    clear[:, 1] = orig[:, 1] + c.rack_y + self._start_off[:, 1] + rec[:, 1]
                    # robotiq: +5 cm — the maneuver's fast drive at the entrance
                    # altitude pressed the low-hanging jammed bottle into the
                    # rack at ~35 N (over glass break; smokes 47/48). Hover
                    # clear; the gentle post-recovery burst does the touchdown.
                    _rc_dz = 0.05 if self.cfg.gripper == "robotiq_2f140" else 0.0
                    clear[:, 2] = orig[:, 2] + c.forge_approach_z + self._tcp_dz + _rc_dz
                    # keep the c.lam carrot. A 0.015 slow-down (tried against a
                    # branch-flip theory after loop 10's fling) BROKE the
                    # direct attempt-1 seat: smokes 61-63 (c.lam) seated on
                    # the first post-recovery attempt every time; smokes 64+
                    # (0.015) never did — the maneuver no longer completes its
                    # correction within its duration. The loop-10 fling was
                    # the staged re-approach (reverted separately), not this.
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
            # GEAR PORT: 1200 (bottle: 300). The variable-kp ori stiffness is
            # an ACCELERATION gain — at 300 it delivers ~1-2 Nm against the
            # hanging 2F-140's 2-3.6 Nm unmodeled moment, and the wrist slowly
            # rotates during the low-force abs insertion (debug smoke 3: gear
            # tilt kept growing 11->25 deg with ZERO contact after the press
            # broke away — the rigidly held gear follows the wrist). 1200
            # stopped the drift in the entrance-hold experiments; steel never
            # showed it because the 100 N press anchors the wrist mechanically.
            # 600 = the SETUP value, deliberately (probes 9-11): any gain
            # change at hand-off shifts the wrist's moment-carrying
            # equilibrium (error = M/k) and the "shift" IS the 35 mm walk —
            # 2400 quadrupled the pull toward the stale command and walked
            # every env identically. Plant continuity across the scripted->
            # learned boundary (the kpos-600 lesson again). Contact drops to
            # 300 (chamfer steering) via the _ok_soft_ctr gate below.
            _ok_ins = 600.0 if self.cfg.gripper == "robotiq_2f140" else 110.0
            # robotiq: 300 (not the franka-tuned 110) — the long gripper levers
            # the wrist down as the arm reaches into the cell and the rigidly
            # pinched bottle tilts with it (snap 4: bottle leaning ~40 deg in
            # the cell); a firm wrist keeps the bottle upright through the
            # learned insertion.
            _ok_setup = 400.0
            if self.cfg.gripper == "robotiq_2f140":
                # GEAR PORT: 1200 for the entrance hold. The variable-kp OSC's
                # ori stiffness is an ACCELERATION gain — through the small
                # rotational task inertia it delivers only ~1-2 Nm at 400
                # (stage probe 5: oerr crept 0.19->0.28 rad, qerr axis -x,
                # NO joint anywhere near its clamp), so a ~2 Nm steady roll
                # disturbance from the hanging 2F-140 wins and the tilt
                # displaces the gear off the arrival gate. 3x the gain pulls
                # the equilibrium error under ~0.1 rad; torques stay tiny.
                # (Was written "1200" but the old stiffness clamp made it 600;
                # the clamp is now open for the robotiq, so pin the PROVEN
                # effective value — the setup servo tracked 0.8 mm at this.)
                _ok_setup = 600.0
            if self.cfg.gripper == "robotiq_2f140" and self._rq_centered:
                # POST-RECOVERY setup runs at the LEARNED-phase stiffness, not
                # 400: with the maneuver-exit twist seeding a nonzero error,
                # ori_k 400 under the render pipeline's held-torque stepping is
                # closed-loop UNSTABLE in the orientation channel — take 93's
                # oerr diverged 0.013 -> 1.58 rad with a perfectly anchored
                # static command (both q_des and quat_tgt re-anchored at
                # rec_end), and the saturated wrist wrench escaped through the
                # shoulders as the climbing spiral. 300 (take 92) still
                # diverged — slower, as a wrist-PITCH mode (the 0.214 m TCP
                # lever turns pitch into bottle rise) — though xy then tracked
                # the aim cleanly. 110 is the franka-proven render stiffness;
                # the 40-deg-lean concern behind the robotiq 300 applies to
                # the LEVERED in-cell insertion, not this free-hanging
                # descent. The pre-recovery setup traverse (error ~0) keeps
                # 400.
                _ok_setup = 110.0
            ori_k = torch.where(self._setup_ctr > 0,
                                torch.full((self.num_envs,), _ok_setup, device=self.device),
                                torch.full((self.num_envs,), _ok_ins, device=self.device))
            if self.cfg.gripper == "robotiq_2f140":
                # CONTACT-GATED wrist compliance (probe_rq_insert_plant 3): the
                # free-space 1200 is mandatory (the hanging 2F-140's 2-3.6 Nm
                # moment tilts the gear at anything less) but under contact it
                # REFUSES the chamfer's steering moment — the gear wedges at
                # the bore mouth, forces build, and the PADS slip around the
                # stuck gear instead (wrist walked 145 mm while the gear held
                # dxy 2-5 mm; seat never captured from 0.5-4 mm). In contact
                # the shaft itself supports the gear, so the gravity-moment
                # concern is moot — drop to 300 (franka-funnel-like) while
                # insertion force is present, with a 30-step hold so the gate
                # doesn't chatter on contact bursts.
                if not hasattr(self, "_ok_soft_ctr"):
                    self._ok_soft_ctr = torch.zeros(self.num_envs, dtype=torch.long,
                                                    device=self.device)
                # trigger at 0.5 N (probe 6: light 0.5-0.8 N rim engagement
                # never tripped a 2 N gate and the stiff free-space plant
                # ground the gear out of the pads), and ALSO whenever the gear
                # is geometrically engaged (below shaft-tip + 2.5 mm) — force
                # can read ~0 inside the bore mouth.
                # contact-only gate — NO altitude term: probe 8 showed a
                # below-tip altitude gate nerfs the free-space lateral hold
                # exactly during the final approach (aim drifted 2.3 -> 6 mm
                # before contact and the gear shed off the tip edge).
                self._ok_soft_ctr = torch.where(
                    (self._cf_insert > 0.5) & (self._setup_ctr == 0),
                    torch.full_like(self._ok_soft_ctr, 30),
                    (self._ok_soft_ctr - 1).clamp(min=0))
                # NO contact ori softening (probe 12): on this plant the ori
                # error is the gravity-residual carrier — ori 300 DOUBLES the
                # lean exactly when verticality decides the insertion, and a
                # persistent ~1 N in-bore kiss kept the window alive until
                # the tilted bore rested on the tip (gripswing ratchet 34->52
                # mm, z stall). Funnel compliance comes from the in-contact
                # xy rate limit instead; the wrist stays at 600 throughout.
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
            if self.cfg.gripper == "robotiq_2f140":
                # ARRIVAL-ARMED SEAT: when the EE reaches the stage-2 hand-off pose
                # (or setup is about to expire — fallback so the seat never skips),
                # start the seat counter. While it runs the ARM FREEZES and the
                # setup countdown PAUSES (a moving pad frame or squeeze-under-
                # acceleration ejects the bottle; the probe always kissed+squeezed
                # on a static arm). Setup resumes afterwards to hold the hand-off
                # until the policy goes live.
                # POST-RECOVERY RE-CONTACT: when a recovery maneuver completes,
                # re-run the contact-terminated approach descent (aimed at the
                # TRUE cell center — the first recovery clears the seeded fault)
                # so the policy always re-engages FROM CONTACT, the only regime
                # it behaves in. Scripted approach positioning, honestly labeled
                # — the in-contact manipulation stays the policy's.
                _rec_end = (self._rq_rec_prev > 0) & (self._rec_steps == 0)
                if bool(_rec_end[0]):
                    # deferred regrasp seat: NOT fired here — the maneuver-exit
                    # pose is wherever the lift left the hand (9 mm off the
                    # shaft in the trace2 run) and the seat pins the gear ON
                    # the shaft axis, so squeezing here bakes a matching
                    # in-hand offset into the new grip. The pend block below
                    # (before arm_seat) waits for the hand to center over the
                    # pin site AND sink to the hand-off band, then fires.
                    self._rq_rimz[:] = -1.0
                    self._rq_pressctr[:] = 0
                    self._rq_calmctr[:] = 0   # swing-settle: re-settle after every maneuver
                    self._rq_need_settle[:] = True
                    self._rq_settlewait[:] = 0
                    # RE-SNAPSHOT the null-space posture target at the CURRENT
                    # joints: apply_recovery pins the elbow branch at p_gain 80
                    # toward _joint_centers (the OSC-handoff posture). Under the
                    # render pipeline, physics steps run BETWEEN control updates,
                    # so the held posture torque decorrelates from the evolving
                    # configuration and the projection leaks into task space —
                    # dragging the arm across joint space toward the handoff
                    # posture traced the takes-96/97 outward spiral (EE moved +y
                    # from the first post-release substep while the aim carrot
                    # demanded -y, and z tracked its 8 mm/substep command at
                    # 0.2 mm/substep — the pull out-torqued the 15 N task P).
                    # Pinning the posture we are IN keeps the branch-holding
                    # benefit (smoke 79/80's calibrated 80) with ZERO initial
                    # pull; the closed-loop base-aim absorbs any branch offset.
                    if self._joint_centers is not None:
                        self._joint_centers = self._robot.data.joint_pos[
                            :, self._arm_ids].clone()
                    # RE-ANCHOR the orientation command at the REACHED quat —
                    # the same rule this file already applies at the other two
                    # regime handoffs (OSC-init flip, seat->OSC handoff). The
                    # maneuver exits with the live wrist ~0.1 rad off the static
                    # q_des; at ori_k 400 anything past 12/400 = 0.03 rad
                    # SATURATES the 12 Nm wrist joints, and under the render
                    # pipeline's held-torque stepping the saturated wrist loses
                    # ground every interval — oerr ratcheted 0.095 -> 2.25 rad
                    # (take 95) and the OSC pushed the growing orientation
                    # wrench through the SHOULDERS: the takes-95/96/97 outward
                    # spiral that flung the bottle. Re-anchoring zeroes the
                    # error — and re-anchor the slerp DESTINATION too (take 94:
                    # re-anchoring q_des alone woke the previously-inert slerp,
                    # which marched the command back toward the handoff quat at
                    # 0.003 rad/substep — faster than the wrist can physically
                    # slew against the 80-damping drives at its 12 Nm effort
                    # clamp (~0.0025 rad/substep), so the error ratcheted 0.027
                    # -> 1.36 rad and the spiral returned). Hold-what-you-
                    # reached, BOTH halves, exactly like the seat->OSC handoff:
                    # the ~0.03-0.1 rad maneuver-exit twist is within what the
                    # funnel ride and the policy (trained at ori_k 110) absorb.
                    # GEAR PORT: DO NOT re-anchor — probe-11 rule wins here.
                    # The standing ori error is LOAD-BEARING (error = M/k for
                    # the ~2-3.6 Nm unmodeled wrist moment). Re-anchoring
                    # zeroes it, so the wrist re-earns the equilibrium error
                    # after every maneuver, swinging the rigidly-held gear
                    # 0.25 m x ~0.13 rad ≈ 30-40 mm along (-x,+y) EXACTLY as
                    # the policy takes over — every post-recovery attempt in
                    # smokes 1-5 pressed at (0.425, 0.151), 40 mm off-shaft,
                    # regardless of a perfect level/centered re-seat. The
                    # bottle takes-94/95 oerr ratchet this guarded against is
                    # a render-pipeline (held-torque stepping) artifact; the
                    # gear runs ori_k 1200 (small standing error) and the
                    # smoke/eval pipeline steps physics normally. (The two
                    # re-anchor lines are deleted, not gated: this env is
                    # gear-only; the bottle env keeps its own copy.)
                    # gravity-integrator ceiling for the re-descent (see the
                    # downward-only clamp in the ag block)
                    self._rq_ag_cap = self._rq_ag.clone()
                    # hold point = the maneuver's exit pose, snapshotted ONCE —
                    # a fixed target gives the OSC real restoring stiffness
                    # against the maneuver's residual arm momentum (holding at
                    # the live ee is zero-error = zero force, take 97 drift)
                    self._rq_settle_pos = ee_pos_w.clone()
                    # (do NOT release _rq_desc here to re-run the staged
                    # up-over-down approach: the full-speed transport swing
                    # with the bottle in the marginal pinch FLUNG it off the
                    # table under render tracking — loop 10. The re-approach
                    # stays the short hover-descend from the maneuver's
                    # entrance hover.)
                    # FREEZE the base-aim HERE, once per recovery (render loop
                    # 12): recomputing `ee + (rack - bottle)` from the LIVE
                    # bottle every substep closes a feedback loop through the
                    # held bottle's pendulum swing — swing -x, carrot +x —
                    # which PUMPS the swing at resonance under render tracking
                    # (smoke sag damps it). The offset is constant while the
                    # bottle is held, so the falling-edge snapshot is the same
                    # aim without the loop.
                    self._rq_freeze_aim(ee_pos_w, orig)
                    # ARM the centered re-approach ON THE FALLING EDGE (this
                    # arming lived under `if self._rq_centered:` for render
                    # loops 3-15 — a dead gate, since only this code ever sets
                    # it True. The whole centered machinery — aim override,
                    # ±0.04 fence, 3.0 N rim gate, the 500-substep re-descent —
                    # silently never engaged in ANY of those takes.)
                    self._rq_dest_dx = 0.0   # GEAR PORT: top-down grip — the EE
                    # aims the gear directly (bottle: -tcp for the side grip)
                    self._jam_on[:] = False
                    # (no static bias counters here anymore — the centered
                    # re-approach below base-aims CLOSED-LOOP on the measured
                    # bottle position, which absorbs the persistent P-push the
                    # old +x/-y offsets only approximated for the rotate_align
                    # flow; the retract flow drifts differently and kept
                    # re-hitting the rim, ratcheting the bottle out of the pinch)
                    self._rq_centered = True
                    self._setup_ctr = torch.maximum(
                        self._setup_ctr, torch.full_like(self._setup_ctr, 500))
                    import os as _os7
                    if _os7.environ.get("RQ_TRACE") == "1":
                        print("      [rq-recontact] recovery done -> centered "
                              "approach re-descent (500)", flush=True)
                # NO live per-substep aim refresh. The old justification ("live
                # aim ran through render loops 4-9 without the loop-12 pendulum
                # pumping") was VOID — the dead centered-arming gate meant loops
                # 4-15 never engaged this code path at all. The first real render
                # test of the live refresh (take 99, 2026-07-06) reproduced the
                # loop-12 fling exactly: ee + (rack − bottle) recomputed every
                # substep closes a feedback loop through the held bottle's
                # pendulum swing under render tracking (smoke sag damps it, so
                # smokes pass either way) and pumped the arm across the room
                # (ee x +0.31 → −0.61, bottle flung through the floor). The
                # falling-edge snapshot in the rec_end block above is the aim.
                self._rq_rec_prev = self._rec_steps.clone()
                dest = torch.stack([orig[:, 0] + c.rack_x + self._start_off[:, 0]
                                    + self._rq_dest_dx,
                                    orig[:, 1] + c.rack_y + self._start_off[:, 1],
                                    orig[:, 2] + c.forge_approach_z + self._tcp_dz], dim=-1)
                if self._rq_centered:
                    # POST-RECOVERY centered re-approach: base-aim — put the EE
                    # where the bottle base lands on the cell center (same
                    # base-aim as the wedge staging). Replaces the static bias
                    # counters: the arm's persistent P-push differs per
                    # maneuver (rotate_align vs retract), so open-loop offsets
                    # calibrated on one flow re-hit the rim on the other —
                    # each rim press ratchets the neck through the pads until
                    # the bottle drops (render loop 2). FROZEN at the recovery
                    # falling edge (rec_end): the per-substep live version
                    # closed a feedback loop through the held bottle's
                    # pendulum swing and pumped it to a fling (loop 12).
                    dest[:, :2] = self._rq_aim
                near = (ee_pos_w - dest).norm(dim=-1) < 0.03
                self._rq_des_z = appr[:, 2].clone()   # unclamped altitude (z integrator)
                # HOVER-LIFT gate: after the OSC handoff, hold xy AT the grasp
                # spot until the z integrator has learned the missing lift (z
                # within 2 cm for 30 substeps). Carrying before convergence sank
                # the bottle into the rack wall at 25-30 N (smokes 15/16).
                if (not self._rq_drive_mode) and (not self._rq_lift_ok) \
                        and self._setup_ctr[0] > 0:
                    _step_xy = (self._rq_hover_xy - ee_pos_w[:, :2]).clamp(-0.015, 0.015)
                    target = torch.cat([ee_pos_w[:, :2] + _step_xy, target[:, 2:3]], dim=-1)
                    if abs(float(appr[0, 2] - ee_pos_w[0, 2])) < 0.02:
                        self._rq_liftctr += 1
                    else:
                        self._rq_liftctr = 0
                    if self._rq_liftctr >= 90:   # 0.75 s settled — outlives the
                        self._rq_lift_ok = True  # P-transient that fooled smoke 17
                        import os as _os4
                        if _os4.environ.get("RQ_TRACE") == "1":
                            print(f"      [rq-lift ok] ag={float(self._rq_ag[0]):+.3f} "
                                  f"ee={[round(float(v),3) for v in ee_pos_w[0]-orig[0]]}",
                                  flush=True)
                # DEFERRED REGRASP SEAT (recovery): fire only once the hand is
                # CENTERED over the pin site (<4 mm — the pin teleports the
                # gear onto the shaft axis, so an off-center hand bakes that
                # offset into the new grip: gxy 16-22 mm, tilt creep, marginal
                # bite in the absfix smokes) AND low enough that the pinned
                # gear lands inside the arrival hand-off gate (gz < 0.465 ⇔
                # ee < ~0.705 — the scripted re-descent cannot reliably sink
                # the weak-z plant afterwards; the POLICY does the descent,
                # exactly like the episode-start hand-off). z-nudge below
                # keeps the maneuver's lift residual sinking meanwhile.
                # Fire-anyway cap 300: a pend that can't center must not
                # starve the attempt loop (is_failure is gated on pend).
                if bool(self._rq_regrasp_pend[0]) \
                        and int(self._rec_steps[0]) == 0 \
                        and int(self._rq_seatctr[0]) == 0:
                    self._rq_pendctr[0] += 1
                    # sink toward the hand-off band, gated on the REALIZED ee
                    # (the weak-z plant carries a ~3 cm standing error, so a
                    # command-side floor never realizes low enough and the fire
                    # gate starves). An unfloored nudge drove the tilted held
                    # gear into the shaft while pend suppressed jam detection
                    # (absfix3: 8/10 breaks, 121 N) — stop sinking once the
                    # realized ee is in band, back off on any contact (the
                    # 18°-tilted rim hangs ~5 mm low and can graze the tip).
                    _eez_now = float(ee_pos_w[0, 2] - orig[0, 2])
                    if float(self._cf_insert[0]) > 2.0:
                        target[0, 2] = ee_pos_w[0, 2] + 0.010
                    elif _eez_now > 0.688:
                        target[:, 2] = torch.minimum(
                            target[:, 2], ee_pos_w[:, 2] - 0.02)
                    else:
                        target[0, 2] = ee_pos_w[0, 2]
                    _pin0 = torch.stack(
                        [orig[0, 0] + c.rack_x + self._start_off[0, 0],
                         orig[0, 1] + c.rack_y + self._start_off[0, 1]])
                    _exy0 = float((ee_pos_w[0, :2] - _pin0).norm())
                    _eez0 = float(ee_pos_w[0, 2] - orig[0, 2])
                    _stl0 = bool(self._rq_settling[0]) \
                        if hasattr(self, "_rq_settling") else False
                    # fire gate 0.690, not 0.705: at 0.703 the pinned gear
                    # lands at z 0.460 — 15-20 mm ABOVE the clean-handoff band
                    # (0.430-0.446), and from that never-trained free hover
                    # the policy mean saturates toward the fence corner (the
                    # smoke 1-8 walk to (0.421,0.160); anchor + obs verified
                    # clean, trace8). At 0.690 the gear lands ~0.445 =
                    # mid-band, the state the policy descends from 256/256.
                    if ((_exy0 < 0.004 and _eez0 < 0.690 and not _stl0
                         and float(self._cf_insert[0]) < 1.0)
                            or int(self._rq_pendctr[0]) > 300):
                        import os as _os9
                        if _os9.environ.get("RQ_TRACE") == "1":
                            print(f"      [rq-pendseat] fired at exy="
                                  f"{_exy0*1000:.1f}mm eez={_eez0:.3f} "
                                  f"pend={int(self._rq_pendctr[0])}",
                                  flush=True)
                        self._rq_regrasp_pend[0] = False
                        self._rq_pendctr[0] = 0
                        # +120 substeps of pinned open-settle BEFORE the kiss:
                        # the recovery seat ENTERS with fingers closed (~0.6)
                        # on the tilted gear, and at the staging timing the
                        # kiss phase began while the pads were still mid-open
                        # (ang 0.31 at ctr 220, trace3) — the squeeze then
                        # stalled 0.604-0.612 vs staging's 0.569 (pads on the
                        # wrong band) and the gear slowly PIVOTED out of the
                        # marginal bite ([rq-c]: gear 0.465->0.471 while the
                        # ee held ±0.5 mm). Phases are counter thresholds, so
                        # extra counter = pure open time at the start.
                        self._rq_seatctr[0] = self._RQ_SEAT_HI + 120
                        self._rq_reaim_pend[0] = True
                        self._rq_seat_hold[0] = ee_pos_w[0]
                        # the seat IS the settle (390 substeps pinned at zero
                        # velocity) — clear the rec_end settle request and
                        # re-snapshot settle_pos to the CENTERED pose. A live
                        # settle hold at the maneuver-exit settle_pos was the
                        # last setup command at handoff, and the policy anchor
                        # inherits it via _last_cmd_target (command
                        # continuity): the arm then held the displaced pose —
                        # the 40 mm (-x,+y) fence-corner walk of smokes 1-7.
                        self._rq_need_settle[0] = False
                        self._rq_settlewait[0] = 0
                        self._rq_settle_pos[0] = ee_pos_w[0]
                        # cooldown covers seat window + a full detector window
                        # AFTER handoff, all in SUBSTEP units (cooldown, the
                        # seat counter and the sig histories all tick per
                        # substep — the //decimation version expired before
                        # the window even ended and the detector churned a
                        # cosmetic hover at handoff, trace3)
                        # ... + 340: protect ~170 ctl steps of POLICY search
                        # after handoff. The healthy funnel thread is a
                        # sustained 10-20 N press with slow descent over
                        # 100-300 steps; a detector window right at handoff
                        # read it as a wedge and yanked the gear MID-SEAT
                        # (smoke 9 ep1 s550: 11.7 N on-axis press -> regrasp
                        # churn). A policy that hasn't seated after the
                        # protected window is honestly stuck.
                        self._jam_cooldown[0] = max(
                            int(self._jam_cooldown[0]),
                            self._RQ_SEAT_HI + 120 + self.cfg.jam_window + 340)
                # The seat arms the moment the PRE-DRIVE completes: the servoed
                # drive pose IS the hand-off (base move), and the whole window
                # runs on the stiff joint drives — no arrival/fallback gating.
                arm_seat = (self._rq_seatctr < 0) & (self._setup_ctr > 0) \
                           & (self._rq_predrive == 0)
                if bool(arm_seat[0]):
                    import os as _os
                    if _os.environ.get("RQ_TRACE") == "1":
                        from isaaclab.utils.math import quat_apply as _qa
                        _hq = self._robot.data.body_quat_w[:1, self._ee_idx]
                        _ax = _qa(_hq, torch.tensor([[0.0, 0.0, 1.0]],
                                                    device=self.device))[0]
                        print(f"      [rq-arm] via={'PREDRIVE(near)' if bool(near[0]) else 'PREDRIVE(off-dest)'} "
                              f"setup={int(self._setup_ctr[0])} "
                              f"ee={[round(float(v),3) for v in ee_pos_w[0]-orig[0]]} "
                              f"dest={[round(float(v),3) for v in dest[0]-orig[0]]} "
                              f"hand_axis={[round(float(v),3) for v in _ax]}", flush=True)
                self._rq_seatctr = torch.where(arm_seat,
                                               torch.full_like(self._rq_seatctr, self._RQ_SEAT_HI),
                                               self._rq_seatctr)
                # snapshot the hold pose on the arming edge: a FIXED target gives
                # the OSC real restoring stiffness through the squeeze (live-ee
                # hold = zero force; the arm crept 2 cm and the bite failed)
                self._rq_seat_hold = torch.where(arm_seat.unsqueeze(-1),
                                                 ee_pos_w, self._rq_seat_hold)
                seat_on = self._rq_seatctr > 0
                target = torch.where(seat_on.unsqueeze(-1), self._rq_seat_hold, target)
                self._rq_seatctr = torch.where(seat_on, self._rq_seatctr - 1, self._rq_seatctr)
                # settle gate pauses the countdown too — the re-descent window
                # must not be consumed while waiting out the pendulum swing
                _hold = seat_on | getattr(self, "_rq_settling",
                                          torch.zeros_like(seat_on))
                self._setup_ctr = torch.where(_hold, self._setup_ctr,
                                              (self._setup_ctr - 1).clamp(min=0))
                # deferred-regrasp aim refresh: the rec_end freeze saw the
                # PRE-seat gear; re-snapshot now that the seat has re-centered
                # the part in-hand and it hangs freely again
                if bool(self._rq_reaim_pend[0]) and int(self._rq_seatctr[0]) == 0:
                    self._rq_freeze_aim(ee_pos_w, orig)
                    self._rq_reaim_pend[0] = False
            else:
                # GEAR PORT: keep the scripted setup alive (floor 1) until the EE
                # has ARRIVED at the approach pose — hand off to the learned
                # policy from the entrance, not from mid-travel. Hard timeout
                # (900 substeps) releases a reach-limited arm.
                # Arrival judged on the GEAR (what the policy inherits), not the
                # EE: with gear-aim the appr target moves with the grip offset and
                # OSC gravity sag keeps |ee-appr| ~1 cm forever (probe_aim: setup
                # never ended). Gear within 8 mm of the shaft, at entrance height.
                _fk_gxy = (self._obj.data.root_pose_w[:, :2]
                           - torch.stack([orig[:, 0] + c.rack_x + self._start_off[:, 0],
                                          orig[:, 1] + c.rack_y + self._start_off[:, 1]],
                                         dim=-1)).norm(dim=-1)
                _fk_gz = self._obj.data.root_pose_w[:, 2] - orig[:, 2]
                # FORGE-faithful staging noise: the paper initializes insertions
                # with few-mm pose noise at the entrance. Our 8 mm arrival left the
                # policy pressing the bore mouth on the shaft tip ~1 cm off — a
                # pressed gear cannot slide (19 N x mu 1.0 friction vs ~1 N lateral
                # OSC authority), so alignment was undiscoverable (two plateaus:
                # dxy 0.017 and 0.011). Funnel capture radius is ~1-1.5 mm; from
                # 3-4 mm force-guided search finds the bore.
                _fk_arr = (_fk_gxy < 0.003) & (_fk_gz < 0.465)
                if not hasattr(self, "_fk_setup_age"):
                    self._fk_setup_age = torch.zeros_like(self._setup_ctr)
                self._fk_setup_age = torch.where(
                    self._setup_ctr > 0, self._fk_setup_age + 1,
                    torch.zeros_like(self._fk_setup_age))
                _fk_floor = torch.where(
                    _fk_arr | (self._fk_setup_age > 900),
                    torch.zeros_like(self._setup_ctr),
                    torch.ones_like(self._setup_ctr))
                # LATCHED: only envs still in setup get the floor — once handed
                # off (ctr==0) the learned policy keeps control even if the EE
                # later drifts out of the arrival ball (probe: 0/1 flapping).
                self._setup_ctr = torch.where(
                    self._setup_ctr > 0,
                    torch.maximum(self._setup_ctr - 1, _fk_floor),
                    self._setup_ctr)
            if self.cfg.gripper == "robotiq_2f140":
                # snapshot the EXACT command the OSC receives — the anchor
                # inits from the last setup step's value (continuity by
                # construction; the appr variable mid-function carries the
                # -0.10 descent carrot / misses aim corrections, probe 12)
                self._last_cmd_target = target.clone()
            return target, ori_k

        def _forge_get_observations(self) -> dict:
            r = self._robot
            jp = r.data.joint_pos[:, self._arm_ids]
            jv = r.data.joint_vel[:, self._arm_ids]
            ee_p = r.data.body_pos_w[:, self._ee_idx] - self.scene.env_origins
            ee_q = r.data.body_quat_w[:, self._ee_idx]
            if self.cfg.gripper == "robotiq_2f140":
                # CROSS-EMBODIMENT OBSERVATION SHIM (policy unchanged): the policy
                # was trained on the franka's proprioception; the robotiq's joint
                # pose / EE frame are far off that manifold and the policy reacts
                # with an up-and-away retreat (smoke 26 vs the franka reference).
                # Present the FRANKA-equivalent view of the same physical state:
                #  * ee_p -> the VIRTUAL franka hand (grasp point - 0.067 along
                #    the approach axis; the robotiq hand sits 0.214 behind it),
                #  * ee_q / jp / jv -> the franka hand-off reference (constant;
                #    the franka's own values barely move during the insertion).
                # base_to_goal / obj_up / ft stay PHYSICAL — the task features.
                # GEAR PORT: virtual panda hand = the point 0.135 above the GEAR
                # (franka hand = gear + mug_grip_z 0.032 + TCP 0.103), anchored
                # to the gear pose DIRECTLY. The old ee+0.109*approach construct
                # inherited the wrist-deflection swing — the virtual hand sat a
                # constant ~19 mm (+y) off the franka-trained hand-gear geometry
                # and the deterministic mean faithfully landed the gear 19 mm
                # off the bore (raw franka ckpt, rq_v4 it400/it600: off std
                # 1.6-3 mm, PPO barely moved it in 600 its — no successes, no
                # gradient). Gear-anchored, the hand-gear relation is franka-
                # EXACT by construction and deflection-invariant; no info the
                # franka policy didn't have (real ee_p composed the same way).
                # ee_q / jp = the franka gear policy's mid-insertion reference
                # (gear_states.npz k=170 + pybullet FK; the franka's own values
                # barely move during the insertion).
                ee_p = (self._obj.data.root_pose_w[:, :3] - self.scene.env_origins
                        + torch.tensor([0.0, 0.0, 0.135], device=self.device))
                ee_q = torch.tensor([0.002, 0.992, 0.113, -0.047], device=self.device) \
                    .expand(self.num_envs, 4)
                jp = torch.tensor([0.099, -0.234, 0.141, -1.966, 0.040, 1.639, 0.789],
                                  device=self.device).expand(self.num_envs, 7)
                jv = torch.zeros_like(jv)
            ft = self._contact_wrench_6d()
            base_w = self._obj.data.root_pose_w[:, :3]
            base_to_goal = self._forge_goal_w() - base_w                 # (N,3) relative goal
            obj_up = torch.bmm(matrix_from_quat(self._obj.data.root_pose_w[:, 3:7]),
                               torch.tensor([0., 0., 1.], device=self.device).view(1, 3, 1)
                               .expand(self.num_envs, 3, 1)).squeeze(-1)
            fcmd = self.f_cmd_norm()
            # BOUNDED obs: with forge_no_term a blown-up env (OSC runaway, void-
            # falling bottle) keeps feeding the nets until truncation — unclamped
            # ee_p/ft/base_to_goal reached 300 m / 2e7 N in the 256-env run and
            # would NaN the policy. Bounds are transparent for nominal episodes
            # (ee_p ~0-1.2 m, ft < 30 N, |base_to_goal| < 1 m).
            ee_p = ee_p.clamp(-2.0, 2.0)
            ft = ft.clamp(-100.0, 100.0)
            base_to_goal = base_to_goal.clamp(-2.0, 2.0)
            # 7+7+3+4+6+3+3+1 = 34
            return {"policy": torch.cat([jp, jv, ee_p, ee_q, ft, base_to_goal, obj_up, fcmd], dim=-1)}

        def _forge_get_rewards(self) -> torch.Tensor:
            c = self.cfg
            cf = self._insertion_force()       # penalise the TRUE insertion force (matches break gate)
            base = self._obj.data.root_pose_w[:, :3] - self.scene.env_origins
            goal = self._forge_goal_w() - self.scene.env_origins
            d = base - goal
            # clamp: beyond 2 m the bottle is gone either way; an unclamped dist
            # follows a void-falling bottle to -300 m (forge_no_term keeps it
            # falling all episode) and wrecks the value targets.
            dist = d.norm(dim=-1).clamp(max=2.0)
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
            # GEAR: ANISOTROPIC alignment shaping — a second PBRS potential on the
            # xy offset alone (2x keypoint weight, ungated so it telescopes
            # exploit-free). Escapes the shaft-tip wedge local optimum: under the
            # isotropic 3D term, pressing at 3-5 mm off pays nearly as well as
            # aligning, and the strict-criterion run sat there for 290 iterations
            # (21 N tip press, succ 0). Lateral progress now pays 3x total.
            dxy_pot = d[:, :2].norm(dim=-1).clamp(max=0.5)
            valid_xy = self._prev_dxy < 9.0
            shape_xy = torch.where(valid_xy, self._prev_dxy - dxy_pot, torch.zeros_like(dxy_pot))
            self._prev_dxy = dxy_pot.clone()
            r = r + 2.0 * c.keypoint_k * shape_xy * live.float()
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
            # bounded per-step reward: a blown-up env's jvel/force penalties hit
            # -300/step (256-env run, it 65) and wreck the value targets. Nominal
            # steps live in [-3, +52] (success cliff +50); the clamp only bites
            # on physics-explosion garbage.
            return r.clamp(-25.0, 60.0)

        def _forge_get_dones(self):
            c = self.cfg
            cf = self._insertion_force()       # gate breakage on the TRUE bottle↔rack insertion force
            base = self._obj.data.root_pose_w[:, :3] - self.scene.env_origins
            in_cell = ((base[:, 0] - c.rack_x).abs() < 0.005) & ((base[:, 1] - c.rack_y).abs() < 0.005)
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
                # dense retract progress (reward moving the hand away after release).
                # Gated on the bottle still being IN THE CELL (xy): with forge_no_term
                # a flung bottle free-falls for the rest of the episode and "hand to
                # bottle distance grows" pays the +0.1 clamp every step — a fling-farm
                # the 256-env run discovered by it 50 (badrel 38->139, bottles at
                # -300 m). Termination used to cut this off; the gate replaces it.
                self._retract_prog = (self._released & in_cell).float() \
                    * (eo_dist - self._prev_eod).clamp(-0.1, 0.1)
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
            return float(self._budget_env[0].item())

        @property
        def max_steps_per_attempt(self) -> int:
            # franka: flat 240 (the committed demo's calibration — do not change).
            # robotiq: 240 insertion steps ON TOP of the scripted setup traverse +
            # the seat window that pauses it — the FIRST attempt spends all of that
            # in-window (a flat 240 made attempt 0 time out INSIDE the setup and
            # cascade into pointless recoveries).
            if self.cfg.gripper != "robotiq_2f140":
                return 240
            # GEAR PORT staging v2: predrive + seat + hover/handoff (~700
            # substeps), NOT the bottle's full setup window (the arrival
            # fast-forward ends the 4000 window in ~250 env steps).
            return 240 + (self._RQ_PREDRIVE + self._RQ_SEAT_HI + 300) // self.cfg.decimation

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
            if self.cfg.gripper == "robotiq_2f140":
                self._rq_regrasp_pend.zero_()
                self._rq_reaim_pend.zero_()
                self._rq_pendctr.zero_()
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
            in_cell = abs(bx - c.rack_x) < 0.005 and abs(by - c.rack_y) < 0.005
            return bool(in_cell and abs(bz - c.cell_floor_z) < c.insert_depth_tol)

        def is_failure(self) -> bool:
            """Jam: high contact with no descent over the window, near the cell."""
            c = self.cfg
            if self._jam_cooldown[0] > 0 or self._rec_steps[0] > 0:
                return False
            if self.cfg.gripper == "robotiq_2f140" \
                    and bool(self._rq_regrasp_pend[0]):
                # a deferred regrasp seat is still waiting to fire (centering
                # traverse) — not a new jam; the pend fire-anyway cap bounds it
                return False
            hover_ok = False
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
                # a hover only counts while the part is still ABOVE the seat (a
                # seated gear at rest is success-pending, not stuck)
                hover_ok = bz > c.cell_floor_z + 0.01
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
            if peak >= thresh and descent_mm < c.jam_progress_mm:
                return True
            # contactless hover: near the cell, no descent, and essentially NO contact —
            # the shy action mean backing off a slip and floating (jam smoke: 500+ steps
            # at z~0.447, 0.00 N). Just as stuck as a wedge, invisible to the force branch.
            # descent_mm > -10: a window straddling a recovery lift reads strongly
            # NEGATIVE (rising) — that's maneuver settling, not a hover; firing on it
            # churned the whole attempt budget before the re-approach could descend.
            return bool(hover_ok and peak < c.jam_hover_n
                        and -10.0 < descent_mm < c.jam_progress_mm)

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
            # count only MEANINGFUL lateral reversals: sub-0.3 N sensor noise flips sign
            # constantly in free space, and phantom "slip events" would misroute a
            # contactless hover to regrasp instead of retract_and_reapproach
            slips = int(((lx[1:] * lx[:-1] < 0) & (lx[1:].abs() > 0.3)).sum().item()) \
                if lx.numel() > 1 else 0
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
            if self.cfg.gripper == "robotiq_2f140":
                # PIN THE ELBOW BRANCH from the first recovery onward: the
                # render pipeline's faster tracking lets the maneuver swing
                # settle the redundant arm in a posture branch the smoke
                # never visits, and every post-recovery descent then drifts
                # +6-12 cm in y (bottle hovers on the funnel edge below all
                # force gates — 0 seats in ~60 rendered post-recovery
                # attempts across loops 3-14). The construction-time
                # null-space stiffness (15) is too weak to hold the posture;
                # 80 shrank the attempt-1 landing miss from a 0.36 N graze to
                # a 3.36 N press (smoke 79). NON-MONOTONIC: 200 regressed to
                # 0.74 N (smoke 80) — past ~100 the posture torque distorts
                # the task-space landing instead of helping. 80 is the
                # calibrated value. Gains are plain tensor attrs on the
                # controller — safe to retune here.
                _np = torch.tensor(80.0, dtype=torch.float, device=self.device)
                self._osc._nullspace_p_gain = _np
                self._osc._nullspace_d_gain = (
                    2.0 * torch.sqrt(_np)
                    * torch.tensor(self._osc.cfg.nullspace_damping_ratio,
                                   dtype=torch.float, device=self.device))
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
                if self.cfg.gripper == "robotiq_2f140":
                    # DEFER the seat window to the maneuver falling edge (rec_end):
                    # the seat pin teleports the gear onto the shaft axis, and from
                    # an in-contact press that means into OVERLAP with the shaft
                    # (~300 N penetration spikes, fragile breaks). Lift first; the
                    # rec_end block fires the seat in free space.
                    self._rq_regrasp_pend[0] = True
                else:
                    self._warmup[0] = self.cfg.warmup_substeps
                    # the regrasp REMOVES the in-hand offset the frozen gear-aim
                    # was compensating — a stale aim then pushes the re-centered
                    # gear the same distance off the OTHER way (smoke5: persistent
                    # same-bias wedges after every regrasp). Clear it; the rec_end
                    # freeze below will re-measure if a NEW jam follows.
                    self._fk_aim[0] = 0.0
                    # the pre-regrasp hover drifts the HAND ~2 cm off the shaft
                    # and the re-seated gear follows the hand — re-engage the
                    # arrival-gated setup so the descent restarts over the shaft
                    self._start_off[0] = 0.0
                    self._setup_ctr[0] = max(int(self._setup_ctr[0]), 40)
                self._rec_off[0, 2] = 0.4 * c.rec_lift

        def _rq_freeze_aim(self, ee_pos_w, orig, alpha=1.0):
            """Update the post-recovery base-aim (see the rec_end block).

            ee + (rack - bottle) is an absolute ee-target that puts the HELD
            bottle over the cell. alpha=1 snapshots it (recovery falling
            edge); alpha<<1 low-pass tracks it while free-hanging — never
            live per-substep: that closes a feedback loop through the held
            bottle's pendulum swing and pumps it to a fling under render
            tracking (render loop 12). Falls back to the nominal chimney
            center when the pinch is empty (grip joint at free-close) or the
            bottle is off the rack area — chasing a dropped bottle dragged
            the arm across the room (render loop 13).
            """
            c = self.cfg
            _bwf = self._obj.data.root_pose_w[:, :3]
            aim = ee_pos_w[:, :2] + torch.stack(
                [orig[:, 0] + c.rack_x - _bwf[:, 0],
                 orig[:, 1] + c.rack_y - _bwf[:, 1]], dim=-1)
            _rc = torch.stack([orig[:, 0] + c.rack_x,
                               orig[:, 1] + c.rack_y], dim=-1)
            # held-check by bottle-to-EE DISTANCE, not the grip joint: the
            # four-bar drive joint flutters past 0.75 transiently during
            # maneuver swings while still holding (smokes 67-69 aborted on a
            # joint threshold). Distance thresholds must clear the DYNAMIC
            # range too: held base-to-EE is ~0.25 m static and grazes 0.30 in
            # a swing — a 0.30 cut fired at the attempt-2 falling edge and
            # sent the aim to nominal (smokes 71/72). A truly dropped bottle
            # measures >1 m from the wandered arm (loop 13); use 0.45.
            _lost = (((_bwf[:, :2] - _rc).norm(dim=-1) > 0.45)
                     | ((_bwf[:, 2] - orig[:, 2]) < 0.25)
                     | ((_bwf - ee_pos_w).norm(dim=-1) > 0.45))
            # GEAR PORT: nominal aim ON the shaft xy (top-down EE; the bottle's
            # side grip needed the -tcp horizontal pullback here)
            _nom = torch.stack(
                [orig[:, 0] + c.rack_x + self._start_off[:, 0],
                 orig[:, 1] + c.rack_y + self._start_off[:, 1]], dim=-1)
            cand = torch.where(_lost.unsqueeze(-1), _nom, aim)
            self._rq_aim = (1.0 - alpha) * self._rq_aim + alpha * cand

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
            # GEAR PORT: signature history from the INSERTION-ONLY channel — the
            # whole-body cvec/_cf_filt carries the ~10 N hub squeeze, which made
            # every stationary near-cell moment read as a jam (grip-polluted).
            cvec = self._raw_insertion_vec3()                     # (N,3) gear<->base force vector
            _p = self._sig_ptr % self._sig_len
            self._cf_hist[:, _p]    = self._cf_insert
            self._basez_hist[:, _p] = base_z_now
            self._latx_hist[:, _p]  = cvec[:, 0]
            self._laty_hist[:, _p]  = cvec[:, 1]
            self._sig_ptr += 1
            self._jam_cooldown = (self._jam_cooldown - 1).clamp(min=0)
            # ── In-grip slip disturbance (one-shot per episode) ─────────────
            if self.cfg.slip_disturb_mm != 0.0 and self.cfg.forge_mode:
                if not hasattr(self, "_slip_done"):
                    self._slip_done = torch.zeros(self.num_envs, dtype=torch.bool,
                                                  device=self.device)
                    self._slip_armed = (torch.rand(self.num_envs, device=self.device)
                                        < self.cfg.slip_frac)
                _gxy = self._obj.data.root_pose_w[:, :2] - torch.stack(
                    [self.scene.env_origins[:, 0] + self.cfg.rack_x,
                     self.scene.env_origins[:, 1] + self.cfg.rack_y], dim=-1)
                _arm = ((~self._slip_done) & self._slip_armed & (self._setup_ctr == 0)
                        & (self._warmup == 0)
                        & (base_z_now < self.cfg.slip_trigger_z)
                        & (_gxy.norm(dim=-1) < 0.008))
                if bool(_arm.any()):
                    pose = self._obj.data.root_pose_w.clone()
                    if self.cfg.slip_rand_dir:
                        _th = torch.rand(self.num_envs, device=self.device) * 6.2831853
                        _mag = self.cfg.slip_disturb_mm / 1000.0
                        pose[_arm, 0] += _mag * torch.cos(_th[_arm])
                        pose[_arm, 1] += _mag * torch.sin(_th[_arm])
                    else:
                        pose[_arm, 1] += self.cfg.slip_disturb_mm / 1000.0
                    self._obj.write_root_pose_to_sim(pose)
                    self._slip_done = self._slip_done | _arm
                    if bool(_arm[0]):
                        print(f"[slip] gear slipped {self.cfg.slip_disturb_mm}mm in-grip "
                              f"at z={float(base_z_now[0]):.3f}", flush=True)
            _rec_active = self._rec_steps > 0
            self._rec_steps = (self._rec_steps - 1).clamp(min=0)
            _rec_done = _rec_active & (self._rec_steps == 0)
            if _rec_done.any():
                self._rec_off[_rec_done] = 0.0
                self._rec_wiggle[_rec_done] = False
                if self.cfg.gripper != "robotiq_2f140":
                    _rec_done = _rec_done & (self._warmup == 0)  # not mid-regrasp-seat
                    # POST-RECOVERY frozen gear-aim (franka twin of _rq_aim): a
                    # slipped gear rides mm-off IN THE PINCH, the policy aims the
                    # hand as trained and re-jams at the same offset forever (jam
                    # smoke ep4: 3x rotate_align -> abort). Freeze the shaft-minus-
                    # gear xy error at the maneuver end and bias the policy-phase
                    # target by it (frozen, not live — task3 loop-12 lesson).
                    # fk_aim freeze RETIRED (kept at zero): it existed to counter
                    # the in-hand offset, which the regrasp routing now removes
                    # outright — and a residual aim bias DISPLACES the policy's
                    # force-guided search center by its own magnitude (debug
                    # trace: a +3 mm aim pinned a healthy 275-step, 12-15 N
                    # search at x+3-4 mm, outside the 1-1.5 mm capture radius).
                    # Lesson 3 stands: frozen aims fight the closed-loop policy.
                    self._fk_aim[_rec_done] = 0.0
                    # POST-RECOVERY RE-APPROACH (franka twin of the robotiq
                    # _rq_centered descent): the shy deterministic mean never
                    # re-contacts from the recovery hover — a full hover-branch
                    # eval logged 0.00 N peak across 30 recovery attempts. Hand
                    # the re-approach to the SETUP controller: staging offset
                    # cleared (re-alignment IS the recovery's job), arrival-gated
                    # up-over-down to the entrance, budget-frozen on contact —
                    # then the LEARNED policy resumes in its trained in-contact
                    # regime, the same setup->policy hand-off the benchmark
                    # itself uses at episode start. 40 < setup_steps//2 so the
                    # stage split is purely the arrival gate.
                    self._start_off[_rec_done] = 0.0
                    self._setup_ctr[_rec_done] = 40
                    # NOTE the limits of this re-approach: it re-centers the
                    # HAND (useful after the post-slip hover drifts ~2 cm) but
                    # cannot fix the true post-slip anomaly — the slip leaves
                    # the gear TILTED 7-10 deg IN THE GRIP (probe: tilt jumps
                    # 4.6->10.1 deg at the kick and never decays; the position
                    # snaps back, the orientation doesn't). A 7-10 deg bore
                    # cannot thread the 0.4 mm fit band (~3-4 deg max), so the
                    # policy rightly refuses to press. The maneuver that FIXES
                    # it is REGRASP (the warmup seat re-writes the canonical
                    # upright in-grip pose) — the heuristic selector routes a
                    # RECURRING contactless hover there at attempt >= 2.
                    if bool(_rec_done[0]):
                        print("[rec] re-approach: setup engaged (arrival-gated)",
                              flush=True)

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
            if self.cfg.gripper == "robotiq_2f140":
                # Probe-v50 sequence AT THE HAND-OFF: park the bottle out of the
                # arm's path during the setup traverse, then pin it UPRIGHT with its
                # neck at the grasp point (EE + 0.214 along local +z — the same
                # perpendicular side grip as the franka, longer hand) while the pads
                # kiss FROM OPEN and the SQUEEZE to full-close completes WHILE STILL
                # PINNED (kiss leaves 24-16=8 mm clearance; an unpinned bottle would
                # fall during the close travel).
                park = self._rq_seatctr < 0
                seat = self._rq_seatctr > self._RQ_PIN_LO
                carry = park | seat
                if self.cfg.rq_table_pick:
                    # TABLE PICK: no park (the gear RESTS at the pick spot from
                    # reset) and no pin (the table supports the part through the
                    # kiss/squeeze — the close is pure physics). The seatctr
                    # still sequences the finger phases.
                    carry = torch.zeros_like(carry)
            if carry.any() and self.cfg.gripper == "robotiq_2f140":
                R_ee = matrix_from_quat(r.data.body_quat_w[:, self._ee_idx])
                off  = torch.zeros(self.num_envs, 3, device=self.device)
                off[:, 2] = self._grasp_tcp_d
                grasp_c = r.data.body_pos_w[:, self._ee_idx] + torch.bmm(R_ee, off.unsqueeze(-1)).squeeze(-1)
                pose = self._obj.data.root_pose_w.clone()
                pose[carry, 0:3] = grasp_c[carry]
                # GEAR PORT: pin the gear ON THE SHAFT AXIS (+ staging noise),
                # not at the grasp centerline — the residual pose-B droop
                # leaves the grasp point 1-3 mm off, and the converged policy's
                # exploration cannot find the 1.5 mm funnel from there (abs
                # round: min-dist stuck at 2 cm for 105 its). The 1-3 mm
                # in-hand offset this creates is a real gripped-off-center
                # part; the franka's gear-aim staging delivers the same state.
                _sxy = torch.stack(
                    [self.scene.env_origins[:, 0] + self.cfg.rack_x
                     + self._start_off[:, 0],
                     self.scene.env_origins[:, 1] + self.cfg.rack_y
                     + self._start_off[:, 1]], dim=-1)
                pose[carry, 0:2] = _sxy[carry]
                pose[carry, 2] = grasp_c[carry, 2] - self._rq_grip_h
                # traverse: hold the bottle high above the cell, clear of the arm
                pose[park, 0] = self.scene.env_origins[park, 0] + self.cfg.rack_x
                pose[park, 1] = self.scene.env_origins[park, 1] + self.cfg.rack_y
                pose[park, 2] = self.scene.env_origins[park, 2] + self.cfg.transport_z + 0.45
                pose[carry, 3] = 1.0; pose[carry, 4:7] = 0.0
                self._obj.write_root_pose_to_sim(pose)
                vel = self._obj.data.root_vel_w.clone()
                vel[carry] = 0.0
                self._obj.write_root_velocity_to_sim(vel)
            elif carry.any():
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
                if self.cfg.gripper == "robotiq_2f140" and self.cfg.forge_mode:
                    # TELEPORT CONTRACT: the robotiq spawns at the folded USD default
                    # pose (no reset teleport), so the "natural" orientation captured
                    # here is NOT a grasp pose. The hand must end at the probe-approved
                    # PERPENDICULAR side grip: approach = world +x (horizontal), finger
                    # spread = world y (probe_rq_scene QUATONLY dump, servo-converged;
                    # the seat angles 0.218/0.589-stall were calibrated at this pose).
                    # GEAR PORT: the arm now SPAWNS at the top-down grasp pose
                    # (arm-only reset teleport), so hold-what-you-spawned — no
                    # re-orientation trajectory at all. (The bottle needed the
                    # folded->side-grip slerp constant here; the OSC handoff
                    # re-anchors both halves to the reached quat regardless.)
                    self._rq_quat_tgt = ee_quat_w.clone()
                    self._rq_q_des = ee_quat_w.clone()
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
                if self.cfg.gripper == "robotiq_2f140" and self.cfg.forge_mode:
                    # Rate-limited re-orientation from the folded spawn quat to the
                    # side-grip target (~0.17 deg/substep -> the flip completes early
                    # in setup stage 1, no torque standoff with the traverse).
                    q_t = self._rq_quat_tgt
                    dot = (self._rq_q_des * q_t).sum(-1, keepdim=True)
                    q_t = torch.where(dot < 0, -q_t, q_t)
                    q_err = quat_mul(q_t, quat_inv(self._rq_q_des))
                    w_c  = q_err[:, 0].clamp(-1.0, 1.0)
                    ang  = 2.0 * torch.acos(w_c.abs())
                    axis = q_err[:, 1:]
                    nrm  = axis.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                    axis = (axis / nrm) * torch.sign(w_c).unsqueeze(-1)
                    q_step = quat_from_angle_axis(ang.clamp(max=0.003), axis)
                    self._rq_q_des = quat_mul(q_step, self._rq_q_des)
                    self._rq_q_des = self._rq_q_des / self._rq_q_des.norm(dim=-1, keepdim=True)
                    self._ee_quat_des = self._rq_q_des
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
                budget = self._budget_env                             # (N,) F_max
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
            if self.cfg.gripper != "robotiq_2f140" and self.cfg.forge_mode:
                # GEAR PORT: 600 for the whole episode (the robotiq lesson,
                # same numbers): at 400 the rate-limited P tops out at
                # lam x kpos = 10 N/axis — exactly the gravity deficit at this
                # pose, so the setup descent stalls 4 cm above the shaft
                # (guided probes: gz flat at 0.462 under both a fixed low
                # target and a rolling crawl). 600 gives 15 N/axis; keeping it
                # across the scripted->learned boundary keeps the plant
                # identical for the policy.
                _kpos = torch.full_like(_kpos, 600.0)
            if self.cfg.gripper == "robotiq_2f140" and self.cfg.forge_mode:
                # 600 (the OSC's variable-kp ceiling) for the WHOLE robotiq
                # episode: it shrinks the gravity-deficit sag 1.5x, gives the
                # rate-limited P 15 N/axis during setup, and — critically —
                # keeps the plant IDENTICAL across the scripted->learned
                # boundary, so the adapted gravity scale stays in equilibrium
                # (dropping to the franka-trained 400 at the boundary shifted
                # the balance and the arm floated away — smoke 23/snap 3).
                _kpos = torch.full_like(_kpos, 600.0)
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
            import os as _os6t
            if (_os6t.environ.get("RQ_TRACE") == "1"
                    and self.cfg.gripper == "robotiq_2f140"
                    and bool(self._rq_desc[0])
                    and getattr(self, "_holdct", 0) % 200 == 0):
                _sat = (jt[0].abs() >= self._eff_lim - 1e-3).tolist()
                from isaaclab.utils.math import quat_mul as _qm2, quat_inv as _qi2
                _qe = _qm2(self._ee_quat_des[0:1],
                           _qi2(r.data.body_quat_w[0:1, self._ee_idx]))[0]
                print(f"      [rq-tau] jt={[round(float(v),1) for v in jt[0]]} "
                      f"sat={_sat} qerr_axis="
                      f"({float(_qe[1]):+.3f},{float(_qe[2]):+.3f},{float(_qe[3]):+.3f})",
                      flush=True)
            jt = jt.clamp(-self._eff_lim, self._eff_lim)
            _tau_pc = None
            if self.cfg.gripper == "robotiq_2f140" and self.cfg.forge_mode \
                    and not self._rq_drive_mode:
                # ── PAYLOAD COMPENSATION (J^T feedforward) ────────────────────
                # The principled fix the ag integrator approximated in z only:
                # feed the gravity wrench of the un-modeled load — the 2F-140
                # subtree PhysX's gravity comp shorts (loop-jointed) plus the
                # held bottle (not in the articulation at all) — forward
                # through J^T. The MOMENT rows are the point: the un-modeled
                # gravity moment of the long loaded tool is what pinned the
                # wrist at its 12 Nm clamp (the lean/twist/grind + the ori_k
                # instability in every robotiq take). Pure feedforward from
                # PhysX-measured masses about the jacobian's own reference
                # point — NOT a closed-loop estimator (smoke 25: force
                # FEEDBACK through jac rows is positive feedback here), and
                # force+moment TOGETHER (the legacy force-only z row missed
                # its moment half -> the smokes 21-23 parasitic xy drag).
                import os as _os4
                _pc_trace = _os4.environ.get("RQ_TRACE3") == "1"
                _pcg, _pco = float(self.cfg.rq_pc_grip), float(self.cfg.rq_pc_obj)
                _pcm = float(getattr(self.cfg, "rq_pc_moment", 0.0))
                if _pcg != 0.0 or _pco != 0.0 or _pcm != 0.0 or _pc_trace:
                    if self._pc_grip_ids is None:
                        _bn = r.data.body_names
                        self._pc_grip_ids = [i for i, n in enumerate(_bn)
                                             if "panda" not in n]
                        self._pc_grip_m = r.root_physx_view.get_masses() \
                            .to(self.device)[:, self._pc_grip_ids]
                        self._pc_obj_m = self._obj.root_physx_view.get_masses() \
                            .to(self.device).reshape(self.num_envs, -1)[:, 0]
                        if _pc_trace:
                            print(f"      [rq-pc] subtree bodies="
                                  f"{[_bn[i] for i in self._pc_grip_ids]} "
                                  f"m={self._pc_grip_m[0].tolist()} "
                                  f"obj_m={float(self._pc_obj_m[0]):.3f}",
                                  flush=True)
                    _g = 9.81
                    _bp  = r.data.body_pos_w[:, self._pc_grip_ids]      # (N,G,3)
                    _gM  = self._pc_grip_m.sum(-1)                      # (N,)
                    _gcom = (_bp * self._pc_grip_m.unsqueeze(-1)).sum(1) \
                        / _gM.unsqueeze(-1).clamp(min=1e-6)
                    _ow  = self._obj.data.root_pose_w[:, :3]
                    # held gate mirrors the aim-freeze heuristic: distance, not
                    # the fluttering grip joint; off before the seat and after
                    # the learned release.
                    _held = (((_ow - ee_pos_w).norm(dim=-1) < 0.40)
                             & ~self._released & (self._rq_seatctr == 0))
                    # unit-scale wrenches at the EE, world frame (diagnosable)
                    _Fg1 = torch.zeros(self.num_envs, 3, device=self.device)
                    _Fg1[:, 2] = _gM * _g
                    _Fo1 = torch.zeros_like(_Fg1)
                    _Fo1[:, 2] = self._pc_obj_m * _g * _held.float()
                    _Mg1 = torch.cross(_gcom - ee_pos_w, _Fg1, dim=-1)
                    _Mo1 = torch.cross(_ow - ee_pos_w, _Fo1, dim=-1)
                    _Fw = _pcg * _Fg1 + _pco * _Fo1
                    _Mw = (_pcg + _pcm) * _Mg1 + (_pco + _pcm) * _Mo1
                    _wb = torch.cat([quat_apply_inverse(root_quat_w, _Fw),
                                     quat_apply_inverse(root_quat_w, _Mw)],
                                    dim=-1)
                    _tau_pc = torch.bmm(jac_b.transpose(1, 2),
                                        _wb.unsqueeze(-1)).squeeze(-1)
                # Integral z feedforward (see init): learn the missing lift while
                # the SCRIPTED setup owns the arm; hold it frozen afterwards.
                # ADAPTIVE GRAVITY SCALING (supersedes the J-row z feedforward):
                # PhysX's gravity comp is short for this loop-jointed articulation;
                # tau += alpha*grav adds lift in EXACTLY the anti-gravity joint
                # direction — no frame ambiguity, no parasitic lateral push (the
                # J-row wrench dragged xy proportional to the lift force; smokes
                # 21-23 mystery drift). alpha adapts: fast in the anchored hover,
                # slow in the carry, contact-gated trickle in the LEARNED phase
                # (so it never fights an intentional insertion press).
                _in_setup = self._setup_ctr > 0
                if bool(_in_setup[0]):
                    _ref = self._rq_des_z
                    # stop adapting once the rim press is latched — the contact
                    # "error" is intentional, not sag (smoke 30: the integrator
                    # lifted the bottle back off the rim).
                    # GEAR PORT: ALSO freeze once the descent latches — the gear
                    # descent is free-space (no rim to latch on), and adapting
                    # against the descent's z-target dynamics wound ag down to
                    # its -0.3 rail (stage probe 1: zerr -0.019, ag -0.30, the
                    # 30% gravity-torque distortion pushed the arm +y 40 cm).
                    # The hover/carry equilibrium carries over, same rationale
                    # as the learned-phase freeze.
                    _gate = (self._rq_rimz < 0) & (~self._rq_desc)
                    _ki, _ec = (0.15, 0.08) if not self._rq_lift_ok else (0.05, 0.03)
                else:
                    # LEARNED phase: with the ANCHORED policy target the
                    # absolute z reference is BACK (the old freeze predates
                    # the anchor; the EE-relative rationale no longer holds).
                    # The frozen z residual was the hold-trace culprit: the
                    # EE floats ~3.4 mm above its command, the standing
                    # K_z*err downward force torques the kinked arm, and the
                    # wrist creeps to a 7-12 deg coupled equilibrium (the
                    # whole post-hand-off walk). Adapt against the anchor,
                    # contact-gated so it never fights an intentional press.
                    if hasattr(self, "_pol_anchor_set"):
                        _ref = torch.where(self._pol_anchor_set,
                                           self._pol_anchor[:, 2], ee_pos_w[:, 2])
                        # also gate on a ~zero z action: adapting against an
                        # INTENTIONALLY descending anchor is the classic
                        # ag-to-the-rail trap (stage probe 1)
                        _gate = (self._pol_anchor_set & (self._cf_insert < 0.5)
                                 & (self._actions[:, 2].abs() < 0.05))
                        _ki, _ec = 0.05, 0.03
                    else:
                        _ref = ee_pos_w[:, 2]
                        _gate = torch.zeros_like(self._rq_ag, dtype=torch.bool)
                        _ki, _ec = 0.0, 0.01
                _zerr_i = (_ref - ee_pos_w[:, 2]).clamp(-_ec, _ec)
                self._rq_ag = torch.where(
                    _gate, (self._rq_ag + _ki * _zerr_i).clamp(-0.3, 1.2),
                    self._rq_ag)
                if self._rq_centered and self._rq_ag_cap is not None:
                    # POST-RECOVERY: adaptation allowed DOWNWARD only (capped
                    # at the rec_end snapshot). Both prior variants failed one
                    # regime each: free adaptation WOUND UP to +1.2 (2.2x
                    # gravity comp) in take 92's disturbed attempts and pinned
                    # the arm at the ceiling; a full FREEZE (take 91) locked
                    # the smoke's staging-inflated +0.40 in as permanent over-
                    # lift and the re-descent floated above the rim to a 0.0 N
                    # timeout (smokes 90/91). Shedding excess lift is the
                    # integrator's healthy direction in a descent; winding up
                    # is the pathology — allow the former, cap the latter.
                    self._rq_ag = torch.min(self._rq_ag, self._rq_ag_cap)
                if _tau_pc is not None:
                    jt = jt + _tau_pc
                jt = (jt + self._rq_ag.unsqueeze(-1) * grav) \
                    .clamp(-self._eff_lim, self._eff_lim)
                if _pc_trace:
                    # Static-hover wrench fit (env 0): at quasi-static, contact-
                    # free equilibrium the applied torque equals TRUE gravity, so
                    # (jt - grav) = the un-modeled residual; the least-squares
                    # J^T fit recovers it as a base-frame wrench. Compare w_hat
                    # against the unit-scale subtree/bottle predictions to set
                    # cfg.rq_pc_grip / rq_pc_obj. Only trust prints with low vel.
                    self._pc_ct = getattr(self, "_pc_ct", 0) + 1
                    if self._pc_ct % 150 == 5:
                        _res = (jt[0] - grav[0]).unsqueeze(-1)
                        try:
                            _w = torch.linalg.lstsq(
                                jac_b[0].transpose(0, 1), _res).solution.squeeze(-1)
                        except Exception:
                            _w = torch.zeros(6, device=self.device)
                        _sat = (jt[0].abs() >= self._eff_lim - 1e-3).int().tolist()
                        _oerr = float(2.0 * torch.acos(
                            (self._robot.data.body_quat_w[0, self._ee_idx]
                             * self._ee_quat_des[0]).sum().abs().clamp(max=1.0)))
                        _vel = float(ee_vel_b[0, :3].norm())
                        print(f"      [rq-pc] sat={_sat} ag={float(self._rq_ag[0]):+.2f}"
                              f" oerr={_oerr:.3f} vel={_vel:.3f} held={int(_held[0])}"
                              f" w_hat=F({float(_w[0]):+.1f},{float(_w[1]):+.1f},"
                              f"{float(_w[2]):+.1f})M({float(_w[3]):+.2f},"
                              f"{float(_w[4]):+.2f},{float(_w[5]):+.2f})"
                              f" grip1=Fz{float(_Fg1[0,2]):+.1f}"
                              f"M({float(_Mg1[0,0]):+.2f},{float(_Mg1[0,1]):+.2f},"
                              f"{float(_Mg1[0,2]):+.2f})"
                              f" obj1=Fz{float(_Fo1[0,2]):+.1f}"
                              f"M({float(_Mo1[0,0]):+.2f},{float(_Mo1[0,1]):+.2f},"
                              f"{float(_Mo1[0,2]):+.2f})", flush=True)
                # NO closed-loop force estimator through jac_b: applying forces
                # via those rows is POSITIVE feedback for this articulation
                # (smoke 25: saturated +25 N on all axes in 240 substeps and
                # rocket-assisted the drift). The LEARNED phase instead keeps
                # the SAME plant as the scripted phase (kpos 600, frozen ag) —
                # no boundary, so the ag equilibrium simply carries over.
                import os as _os3
                if _os3.environ.get("RQ_TRACE") == "1" \
                        and int(self._setup_ctr[0]) % 200 == 100:
                    print(f"      [rq-ag] setup={int(self._setup_ctr[0])} "
                          f"ag={float(self._rq_ag[0]):+.3f} zerr={float(_zerr_i[0]):+.3f}",
                          flush=True)

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
                # seat-window targets (probe-v50 sequence at the hand-off): pads stay
                # OPEN through the traverse (pending) and the pin-settle beat, KISS
                # onto the pinned bottle, then the full-close squeeze (still pinned
                # until _RQ_PIN_LO — the drive stalls on the neck). Once the window
                # completes (ctr==0) the live close_mask/release logic rules.
                _ctr = self._rq_seatctr
                _scripted = _ctr != 0
                _s_ang = torch.full((self.num_envs,), self._rq_open, device=_d)
                _s_ang = torch.where((_ctr <= self._RQ_KISS_HI) & (_ctr > self._RQ_KISS_LO),
                                     torch.full((self.num_envs,), self._rq_seat, device=_d), _s_ang)
                _s_ang = torch.where((_ctr <= self._RQ_KISS_LO) & (_ctr > 0),
                                     torch.full((self.num_envs,), self._rq_close, device=_d), _s_ang)
                ang_t = torch.where(_scripted, _s_ang, ang_t)
                if self.cfg.forge_release_mode:
                    ang_t = torch.where(self._released,
                                        torch.full((self.num_envs,), self._rq_open, device=_d), ang_t)
                import os as _os
                if _os.environ.get("RQ_TRACE") == "1" and int(self._rq_seatctr[0]) > 0:
                    print(f"      [rqw] sc={int(self._rq_seatctr[0])} "
                          f"ang={float(r.data.joint_pos[0, self._grip_ids[0]]):+.3f} "
                          f"tgt={float(ang_t[0]):.3f} objz={float(self._obj.data.root_pose_w[0,2]):.3f} "
                          f"sep={float((r.data.body_pos_w[0,self._lf_idx]-r.data.body_pos_w[0,self._rf_idx]).norm()):.4f}",
                          flush=True)
                r.set_joint_position_target(
                    ang_t.unsqueeze(-1).expand(-1, len(self._grip_ids)),
                    joint_ids=self._grip_ids)
                if self._rq_drive_mode:
                    # DRIVE MODE (no OSC efforts — they'd fight the drives): the
                    # holding-gain joint drives own the arm from the folded spawn,
                    # through the servoed side-grip pose, through the ENTIRE seat
                    # window (probe regime: stiff, zero sag, level pads).
                    import os as _os2
                    _pd = int(self._rq_predrive.max())
                    if _pd > 0:
                        if _pd % 120 == 0 and _pd < self._RQ_PREDRIVE \
                                and self._rq_servo_on:
                            # probe servo on the settled state (measured on env 0;
                            # demo N=1): yaw err -> j1, x-reach err -> j4 (elbow),
                            # z err -> j2, pitch err -> j6.
                            from isaaclab.utils.math import quat_apply as _qa2
                            import math as _m
                            _bq = r.data.body_quat_w[0:1, self._ee_idx]
                            _ap = _qa2(_bq, torch.tensor([[0.0, 0.0, 1.0]], device=_d))[0]
                            _eep = r.data.body_pos_w[0, self._ee_idx] - self.scene.env_origins[0]
                            _perr = _m.asin(max(-1.0, min(1.0, float(-_ap[2]))))
                            _yerr = _m.atan2(float(_ap[1]), float(_ap[0]))
                            _zerr = float(_eep[2]) - (self.cfg.forge_approach_z + self._tcp_dz)
                            _xerr = float(_eep[0]) - (self.cfg.rack_x
                                                      + float(self._start_off[0, 0])
                                                      + self._rq_dest_dx)
                            if self.cfg.rq_table_pick:
                                # pitch target is STRAIGHT DOWN (top-down grip),
                                # not the bottle's horizontal side grip — the
                                # side-grip formula reads a permanent +90 deg
                                # error for a down-pointing hand and wound j6 to
                                # the clamp corner (probe 4). Deviation = the
                                # axis' radial component; sign: +j6 grew it.
                                _perr = -_m.asin(max(-1.0, min(1.0,
                                                               float(_ap[0]))))
                            if self.cfg.rq_table_pick:
                                # TABLE PICK destination: pads at the grip band
                                # over the table-resting gear. x/z retarget; the
                                # j1 term servos the y POSITION (dy/dj1 ~ reach —
                                # pad yaw is free on a round gear, so hand-axis
                                # yaw is unconstrained here).
                                _cp = _m.cos(self.cfg.rq_pick_dth)
                                _sp = _m.sin(self.cfg.rq_pick_dth)
                                _xerr = float(_eep[0]) - (_cp * self.cfg.rack_x
                                                          - _sp * self.cfg.rack_y)
                                _zerr = float(_eep[2]) - (
                                    self.cfg.table_top_z - 0.005 + self._rq_grip_h
                                    + float(self._grasp_tcp_d))
                                _yerr = 1.6 * (float(_eep[1])
                                               - (_sp * self.cfg.rack_x
                                                  + _cp * self.cfg.rack_y))
                            def _cl(v, lo, hi):
                                return max(lo, min(hi, v))
                            # per-window correction clamps kill the overshoot the
                            # big initial errors caused (smoke 11: x overshot 13 cm
                            # and never pulled back); joint clamps keep the pose
                            # family out of degenerate corners (j2 -> 0, j6 limit).
                            # JOINT CLAMPS keep the servo in the shoulder-back pose
                            # family (probe-like, j2 <= -0.95): chasing z into the
                            # j2->0 vertical-shoulder family lands the OSC near a
                            # singularity of its inertia decoupling and the arm
                            # runs away after handoff (smoke 13 vertical runaway).
                            # joint clamps are POSE-FAMILY bounds: the bottle
                            # family (j2<=-0.95, j4<=-1.55, j6>=1.8) vs the gear
                            # top-down family (probe pose [0.26,-0.45,0,-1.42,0,
                            # 1.44,0.785]) — the gear pose sits OUTSIDE all three
                            # bottle bounds, so a window under the bottle clamps
                            # yanks j2/j4/j6 to the corners (table-pick probe 3:
                            # alien pose at z 0.98, j1 runaway chasing it)
                            if self.cfg.rq_table_pick:
                                _b2, _b4, _b6 = (-0.9, -0.1), (-1.9, -0.95), (1.0, 1.9)
                            else:
                                _b2, _b4, _b6 = (-1.7, -0.95), (-2.9, -1.55), (1.8, 3.6)
                            self._rq_arm_cmd[0] = float(self._rq_arm_cmd[0]) - _cl(0.8 * _yerr, -0.15, 0.15)
                            self._rq_arm_cmd[1] = _cl(float(self._rq_arm_cmd[1])
                                                      + _cl(0.7 * _zerr, -0.15, 0.15), *_b2)
                            self._rq_arm_cmd[3] = _cl(float(self._rq_arm_cmd[3])
                                                      - _cl(0.6 * _xerr, -0.12, 0.12), *_b4)
                            self._rq_arm_cmd[5] = _cl(float(self._rq_arm_cmd[5])
                                                      + _cl(0.8 * _perr, -0.15, 0.15), *_b6)
                            if _os2.environ.get("RQ_TRACE") == "1":
                                print(f"      [rq-servo pd={_pd}] ee={[round(float(v),3) for v in _eep]} "
                                      f"xerr={_xerr:+.3f} zerr={_zerr:+.3f} yaw={_yerr:+.3f} "
                                      f"pitch={_perr:+.3f} cmd={[round(float(v),3) for v in self._rq_arm_cmd]}",
                                      flush=True)
                            # tolerance gate on the LAST window: extend the servo
                            # instead of grasping from a not-converged pose.
                            # pose tolerances: LEVEL pads matter most (taper-wedge);
                            # a few cm of xy/z hand-off offset is fine — the pin is
                            # EE-relative and the policy corrects small leans.
                            _ztol = 0.006 if self.cfg.rq_table_pick else 0.05
                            _xtol = 0.020 if self.cfg.rq_table_pick else 0.06
                            # _ztol 0.006 not 0.012: the slack let the pads grip
                            # 6 mm below the hub centre and the top-heavy gear
                            # leaned 3.5 deg through the carry — right at the
                            # policy's ~3-4 deg cannot-thread refusal boundary
                            # (probe 9: it fled upward from the hand-off)
                            # table pick: the close happens AT the pick pose, so
                            # the pads must be ON the grip band (z) and centred
                            # (x, y-via-j1) before the squeeze; the pin used to
                            # absorb this error and is gone
                            _pdmax = 10 if self.cfg.rq_table_pick else 4
                            if _pd == 120 and self._rq_pd_ext < _pdmax and (
                                    abs(_perr) > 0.04 or abs(_zerr) > _ztol
                                    or abs(_xerr) > _xtol or abs(_yerr) > 0.04):
                                self._rq_predrive += 240
                                self._rq_pd_ext += 1
                                if _os2.environ.get("RQ_TRACE") == "1":
                                    print(f"      [rq-servo] tolerance unmet -> extend "
                                          f"(#{self._rq_pd_ext})", flush=True)
                        self._rq_predrive -= 1
                        if int(self._rq_predrive.max()) == 0 \
                                and _os2.environ.get("RQ_TRACE") == "1":
                            _ee0 = r.data.body_pos_w[0, self._ee_idx] - self.scene.env_origins[0]
                            print(f"      [rq-predrive done] ee={[round(float(v),3) for v in _ee0]} "
                                  f"jpos={[round(float(v),3) for v in r.data.joint_pos[0,:7]]}",
                                  flush=True)
                    if self.cfg.rq_table_pick:
                        # DRIVE-SIDE RETURN TRAVERSE: carry the gripped gear back
                        # to the entrance pose on the STIFF JOINT DRIVES (zero
                        # sag, zero twist-walk). The OSC carry drifted +y 15-30
                        # mm during the settle hover (probes 7-12 — the wrist-
                        # twist disturbance the pinned flow never exposed) and
                        # the press from there wedged the gear out of the pads.
                        # The pick pose is the entrance pose j1-rotated, so
                        # interpolating the COMMAND back to _rq_arm_pose_ent
                        # retraces the arc and lands the OSC handoff exactly
                        # where the pinned flow always handed off.
                        if (self._rq_return < 0 and _pd == 0
                                and int(self._rq_seatctr.max()) == 0
                                and int(self._rq_seatctr.min()) == 0):
                            self._rq_return = 1080
                        if self._rq_return > 0:
                            if self._rq_return % 120 == 0:
                                _stp = (self._rq_arm_pose_ent
                                        - self._rq_arm_cmd).clamp(-0.05, 0.05)
                                self._rq_arm_cmd += _stp
                            self._rq_return -= 1
                    r.set_joint_position_target(
                        self._rq_arm_cmd.unsqueeze(0).expand(self.num_envs, 7),
                        joint_ids=self._arm_ids)
                    if _pd == 0 and int(self._rq_seatctr.max()) == 0 \
                            and int(self._rq_seatctr.min()) == 0 \
                            and (not self.cfg.rq_table_pick
                                 or self._rq_return == 0):
                        # Seat complete, grip established -> hand the arm to the
                        # OSC: zero the gains (parameter write, no body motion),
                        # anchor the nullspace and the rate-limited orientation
                        # command at the REACHED state (a lagging command makes
                        # the OSC yank the wrist backward — smoke-8 collapse).
                        _z = torch.zeros(self.num_envs, 7, device=_d)
                        if self._rq_drive_kp is None:
                            # one-time snapshot of the holding gains so the reset
                            # re-arm can undo this zeroing for episode 2+
                            self._rq_drive_kp = r.data.joint_stiffness[:, self._arm_ids].clone()
                            self._rq_drive_kd = r.data.joint_damping[:, self._arm_ids].clone()
                        r.write_joint_stiffness_to_sim(_z, joint_ids=self._arm_ids)
                        r.write_joint_damping_to_sim(_z + 80.0, joint_ids=self._arm_ids)
                        self._joint_centers = r.data.joint_pos[:, self._arm_ids].clone()
                        self._rq_q_des = r.data.body_quat_w[:, self._ee_idx].clone()
                        # FREEZE the orientation target at the reached quat: the
                        # servo already delivered the level +x side grip, and the
                        # probe quat belongs to a DIFFERENT arm pose family — the
                        # slerp resuming toward it saturates j6 and the growing
                        # error swings the whole arm away (smoke 12, ~300 substeps
                        # after handoff). Hold-what-you-reached is the OSC regime
                        # proven stable since smoke 4.
                        self._rq_quat_tgt = self._rq_q_des.clone()
                        self._rq_drive_mode = False
                        self._rq_lift_ok = False
                        self._rq_liftctr = 0
                        self._rq_hover_xy = r.data.body_pos_w[:, self._ee_idx, :2].clone()
                        if _os2.environ.get("RQ_TRACE") == "1":
                            _ee0 = r.data.body_pos_w[0, self._ee_idx] - self.scene.env_origins[0]
                            print(f"      [rq-osc handoff] ee={[round(float(v),3) for v in _ee0]} "
                                  f"grip_ang={float(r.data.joint_pos[0, self._grip_ids[0]]):+.3f}",
                                  flush=True)
                    return
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
                torch.full((self.num_envs,), 0.015, device=_d),   # squeeze the Ø35.5 mm hub: 0.015/side
                                                                  # = FORGE's 30 mm gripper width (~2.7 mm
                                                                  # interference/side -> friction squeeze)
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
                # RECIPE v3: panda_hand is a container prim, not a body — the robotiq
                # EE body is robotiq_base_link (same frame via the retargeted joint).
                self._ee_idx = bn.index("robotiq_base_link"
                                         if self.cfg.gripper == "robotiq_2f140" else "panda_hand")
                if self.cfg.gripper == "robotiq_2f140":
                    self._lf_idx = bn.index("left_inner_finger")
                    self._rf_idx = bn.index("right_inner_finger")
                    jn = list(self._robot.data.joint_names)
                    # both knuckles driven (symmetric reaction cancel)
                    self._grip_ids = [jn.index("finger_joint"),
                                      jn.index("right_outer_knuckle_joint")]
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
                if len(env_ids) == self.num_envs:
                    # ARM-ONLY teleport to the spawn pose. The teleport contract's
                    # four-bar breakage comes from writing GRIPPER DOFs (the loop-
                    # closure joints don't follow); writing just the 7 arm DOFs
                    # moves the gripper subtree rigidly and the loop stays closed.
                    # Without this, ep-2+ starts from wherever the policy wrecked
                    # the arm — run-4 ep-2 drove j5 past its ±2.9 limit and ep-3's
                    # drives-vs-limit fight exploded the solver (Farm 2e8) with
                    # staging dead for the rest of the run. Teleporting makes every
                    # episode's staging bit-identical for every env.
                    _arm_def = self._robot.data.default_joint_pos[env_ids][:, self._arm_ids]
                    if self.cfg.rq_table_pick:
                        # pick spot = entrance rotated rq_pick_dth about the base:
                        # the SAME proven pose with j1 offset lands the open pads
                        # straddling the table-resting gear
                        _arm_def = _arm_def.clone()
                        _arm_def[:, 0] += self.cfg.rq_pick_dth
                    self._robot.write_joint_state_to_sim(
                        _arm_def, torch.zeros_like(_arm_def),
                        joint_ids=self._arm_ids, env_ids=env_ids)
                    jp[:, self._arm_ids] = _arm_def
                # Re-arm the PRE-DRIVE + DRIVE MODE staging state machine. This
                # state is GLOBAL (python bools + an env-0-servoed arm command),
                # so it is only correct when every env resets together — training
                # guarantees that with forge_no_term (truncation-only dones); the
                # demo resets once at t=0. A partial reset would corrupt the
                # staging of the still-live envs, hence the guard.
                if len(env_ids) == self.num_envs:
                    self._rq_predrive[:] = self._RQ_PREDRIVE
                    self._rq_arm_cmd = self._rq_arm_pose.clone()
                    self._rq_drive_mode = True
                    self._rq_pd_ext = 0
                    self._rq_return = -1
                    # Restore the HOLDING-gain arm drives that the OSC handoff
                    # zeroed — with kp=0 the ep-2+ predrive position targets are
                    # inert and the arm drifts into an alien joint branch (2-ep
                    # smoke: jpos [-0.73,-0.39,0.54,...] vs probe [0,-1.39,0,...],
                    # seat armed 13 cm low, lift integrator railed at +1.2).
                    if self._rq_drive_kp is not None:
                        self._robot.write_joint_stiffness_to_sim(
                            self._rq_drive_kp, joint_ids=self._arm_ids)
                        self._robot.write_joint_damping_to_sim(
                            self._rq_drive_kd, joint_ids=self._arm_ids)
                    # Zero the joint EFFORT targets: the OSC writes them every
                    # step, drive mode never does — so ep-1's final torque command
                    # (arm + gripper squeeze) would keep applying through all of
                    # ep-2's staging (smoke3: arm dragged to j1=-2.6, finger
                    # overdriven to 2.81 rad — four-bar squirt).
                    self._robot.set_joint_effort_target(
                        torch.zeros_like(self._robot.data.joint_pos))
                    self._rq_lift_ok = True
                    self._rq_liftctr = 0
                    self._rq_centered = False
                    self._rq_dest_dx = 0.0   # GEAR PORT (top-down grip; slip inducer, not wedge)
                    self._rq_ag[:] = 0.0
                    self._rq_ag_cap = None
                    self._rq_fup[:] = 0.0
                    self._rq_f[:] = 0.0
                    self._rq_des_z[:] = 0.0
                    self._rq_xyint[:] = 0.0
                    self._rq_prev_ee = None
                    self._rq_prev_cmd[:] = 0.0
                    self._rq_hover_xy[:] = 0.0
                    self._rq_aim[:] = 0.0
                    self._rq_rec_prev[:] = 0
                elif self.cfg.forge_mode:
                    print(f"[rq-reset] WARNING: partial reset ({len(env_ids)}/"
                          f"{self.num_envs}) — global staging state NOT re-armed",
                          flush=True)
            else:
                jp = self._robot.data.default_joint_pos[env_ids].clone()
                # Pre-close the gripper fingers onto the hub's half-width so it starts
                # gripped (then the grip force + friction hold it).
                jp[:, 7:9] = 0.018
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
            # robotiq: NO reset-time warmup — the seat is armed by ARRIVAL at the
            # hand-off during setup (_rq_seatctr: -1 pending). The franka teleports
            # to its anchor at reset, so its warmup seat can run immediately.
            self._warmup[env_ids]      = (0 if self.cfg.gripper == "robotiq_2f140"
                                          else self.cfg.warmup_substeps)
            self._rq_seatctr[env_ids]  = (-1 if self.cfg.gripper == "robotiq_2f140" else 0)
            self._rq_desc[env_ids]     = False
            self._rq_rimz[env_ids]     = -1.0
            self._rq_pressctr[env_ids] = 0
            self._rq_calmctr[env_ids]  = 0
            if self.cfg.gripper == "robotiq_2f140":
                self._rq_need_settle[env_ids] = False
                self._rq_settlewait[env_ids]  = 0
                # deterministic wedge staging: the random start offset moved the
                # jam aim in/out of the funnel's capture radius per episode
                # (smoke 43: a -1 cm draw turned the wedge into a slide-in).
                self._start_off[env_ids] = 0.0
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
            if self.cfg.gripper == "robotiq_2f140":
                if self.cfg.rq_table_pick:
                    # TABLE PICK: spawn resting at the pick spot (gear origin is
                    # 5 mm below the bottom face -> table_top - 0.005; +2 mm drop
                    # margin settles during the predrive window). Real physics
                    # from here on — no park, no pin.
                    _c = math.cos(self.cfg.rq_pick_dth)
                    _s = math.sin(self.cfg.rq_pick_dth)
                    obj_state[:, 0] = (self.scene.env_origins[env_ids, 0]
                                       + _c * self.cfg.rack_x - _s * self.cfg.rack_y)
                    obj_state[:, 1] = (self.scene.env_origins[env_ids, 1]
                                       + _s * self.cfg.rack_x + _c * self.cfg.rack_y)
                    obj_state[:, 2] = (self.scene.env_origins[env_ids, 2]
                                       + self.cfg.table_top_z - 0.005 + 0.002)
                    obj_state[:, 3] = 1.0
                    obj_state[:, 4:7] = 0.0
                else:
                    # STAGING v2: the arm resets AT the entrance pose, so the
                    # default obj spawn (rack, transport_z 0.72) lands INSIDE the
                    # parked gripper — the contact spike (Fins ~7.6) fired the
                    # descent fast-forward within 2 substeps and zeroed the setup
                    # window before the seat could arm (the dead-episode
                    # alternation in ft run 1: closed fingers collide, open ones
                    # don't). Spawn directly at the PARK altitude instead.
                    obj_state[:, 2] = (self.scene.env_origins[env_ids, 2]
                                       + self.cfg.transport_z + 0.45)
            self._obj.write_root_pose_to_sim(obj_state[:, 0:7], env_ids=env_ids)
            self._obj.write_root_velocity_to_sim(obj_state[:, 7:13], env_ids=env_ids)

            # FORGE mode: a setup window drives the EE to a RANDOMIZED approach pose
            # (the policy must then correct the offset + insert from force).
            if self.cfg.forge_mode:
                n = len(env_ids)
                lat = self.cfg.forge_start_lat
                self._start_off[env_ids] = (torch.rand(n, 2, device=self.device) * 2 - 1) * lat
                if self.cfg.forge_start_fixed_x >= 0.0:
                    self._start_off[env_ids, 0] = self.cfg.forge_start_fixed_x
                    self._start_off[env_ids, 1] = self.cfg.forge_start_fixed_y
                # GEAR PORT: no robotiq start_off override — the gear task keeps
                # the FORGE-faithful randomized start noise (forge_start_lat /
                # forge_start_fixed_*) on both grippers. (The bottle zeroed it
                # for deterministic wedge staging; the gear jams via the slip
                # inducer instead.)
                self._setup_ctr[env_ids] = self.cfg.forge_setup_steps
                if hasattr(self, "_slip_done"):
                    self._slip_done[env_ids] = False   # re-arm the slip disturbance
                    self._slip_armed[env_ids] = (
                        torch.rand(len(env_ids), device=self.device)
                        < self.cfg.slip_frac)         # fractional-slip gate reroll.
                    # NOTE: the INSTANTANEOUS armed fraction ratchets above
                    # slip_frac (survivorship: clean episodes finish and re-
                    # roll fast; slipped ones linger to truncation). Each
                    # EPISODE is still armed with probability slip_frac.
                self._fk_aim[env_ids] = 0.0
                # Recovery state is EPISODE state: an episode that truncates
                # mid-recovery otherwise leaks its maneuver into every later
                # episode (a stale _rec_off lift kept the next episodes
                # hovering 11 cm high forever — multi-episode jam eval; task3's
                # single-episode demos never exposed this).
                self._rec_off[env_ids] = 0.0
                self._rec_steps[env_ids] = 0
                self._rec_wiggle[env_ids] = False
                self._jam_cooldown[env_ids] = 0
                self._settle_ctr[env_ids] = 0
                self._best_dist[env_ids] = 9.9
                self._prev_dist[env_ids] = 9.9
                self._prev_dxy[env_ids] = 9.9
                self._min_dist[env_ids] = 9.9

            self._sample_episode(env_ids)

else:
    # Alias so imports work even without Isaac Sim
    FrankaGearInsertEnv = MockGearInsertEnv  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# Gym registration
# ─────────────────────────────────────────────────────────────────────────────
gym.register(
    id="FORGE-GearInsert-v0",
    entry_point="forge_plus.isaac_gear_env:FrankaGearInsertEnv",
    kwargs={"cfg": GearInsertEnvCfg()},
)

gym.register(
    id="FORGE-GearInsert-Mock-v0",
    entry_point="forge_plus.isaac_gear_env:MockGearInsertEnv",
)
