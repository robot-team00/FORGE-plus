# 06 — Force-signature LLM recovery (closed loop, in Isaac)

The proposal's **recovery layer** (§07), now active on the **real contact-rich Isaac env**: when
an insertion jams, the system reads a **text force/contact signature** (no vision, no `F_break`),
a frozen LLM picks a recovery action from a fixed menu, the robot applies it **within the same
`F_max`**, and retries — up to `k_max` attempts. This closes the gap noted in earlier docs (the
recovery code existed but was only exercised in the CPU mock env).

Files: `forge_plus/recovery/recovery_loop.py` (the loop), `forge_plus/llm/recovery_selector.py`
(the LLM call), `forge_plus/isaac_pick_place_env.py` (the env hooks),
`scripts/run_recovery_insertion.py` (the demo), `tests/test_recovery_loop.py` (CPU test).

## 1. Architecture — one loop, every task env

The loop is **task-agnostic**: each task has its own Isaac env, but they all plug into one
orchestrator through a tiny protocol (`RecoveryEnv`). `RecoveryLoop.run(env)`:

```
reset episode
for attempt in range(k_max):
    while steps < max_steps_per_attempt:
        env.step_skill()                 # nominal force-conditioned skill (OSC + phase machine)
        if env.is_success(): return SUCCESS
        if env.is_failure(): break       # jam / over-force / no-progress
    sig = env.failure_signature()        # TEXT force signature — no vision, no F_break
    decision = RecoverySelector.select(sig, f_max, attempt, subphase, gripper)
    if decision.action == "abort": return ABORTED
    env.apply_recovery(decision.action, decision.params)   # F_max NEVER relaxed
return FAIL_NO_ATTEMPTS_LEFT
```

To make a new task env recoverable, implement the `RecoveryEnv` hooks on it — nothing in the loop
or the LLM layer changes. (Scope note: as everywhere in Task 3, this is the **terminal micro-phase**
with **no vision** and **poses assumed known**; see [`05-wine-cellar-insertion.md`](05-wine-cellar-insertion.md) and the root proposal §01.)

## 2. The env hooks (`FrankaPickPlaceEnv`)

| Hook | What it does |
|---|---|
| `step_skill()` | One control step under the nominal zero-action OSC + phase machine. |
| `is_success()` | Geometric seat: base centered in the cell **and** at the cell floor. |
| `is_failure()` | **Jam**: contact at/near the budget with **no net descent** over a short window (`jam_window`), once past `PLACE_DESCEND`. |
| `failure_signature()` | A `ForceSignature` from a contact+pose **ring buffer**: peak/mean axial force, net insertion (mm), rising trend, lateral bias, slip events, contact persistence. **No images, no `F_break`.** |
| `apply_recovery(action, params)` | OSC maneuvers (below). Sets `F_max`-preserving target offsets; **never changes `F_max`**. |

**Recovery primitives** (the fixed menu → OSC motions):
- `retract_and_reapproach` — lift the EE clear of the rim (`rec_lift`), then re-approach; base-aim re-centers on the way back down.
- `wiggle_search` — a small lateral oscillation (`rec_lat`) while lifted, to find the hole.
- `rotate_align` — nudge the bottle **base toward the cell center** (counteracts a lateral wedge) + lift slightly.
- `regrasp` — re-seat the bottle in the gripper (reuse the warmup seat) + lift.
- `abort` — the loop stops before applying anything (never risks the part).

## 3. Force authority — staying under budget (FORGE)

The key faithfulness point: a jam must be caught **without exceeding the force budget** (a glass
bottle would shatter otherwise). The env enforces a **soft force ceiling** in `_apply_action`:
when the filtered contact force reaches ~`0.9·F_max`, the EE **retreats** a few mm instead of
pushing further down. So a wedge **oscillates at the ceiling with no net descent** — exactly the
"at `F_max`, no progress" signal the recovery loop catches, and the contact force stays far below
`F_break`. (Retreat, not freeze: freezing the target z deadlocks once a recovery has cleared the
misalignment; retreating never does.)

## 4. The induced-jam scenario

To exercise the loop, `cfg.jam_dx` (a lateral base-aim error, m) makes the bottle **wedge on the
cell rim**. It is gated by `self._jam_on`, which a recovery clears — modeling a **corrected
approach** (the recovery realigned the insertion). Set `--jam 0.0` for the clean insertion (no
jam, no recovery needed). This is a deliberately simple, controllable failure to demonstrate the
closed loop; richer jam models (angular catch, burr) are future work.

## 5. Verified result

`scripts/run_recovery_insertion.py --jam 0.05 --backend heuristic` (no API key, no GPU vision):

```
F_max = 8.8 N (object budget, from the LLM)      backend = heuristic
attempt 0: failure  [peak_axial=13.7N  net_insert=0.59mm  rising=True]  -> retract_and_reapproach
attempt 1: success
final bottle base (env frame): (0.451, 0.118, 0.429)   cell=(0.45, 0.12, 0.40)
OUTCOME: SUCCESS in 2 attempts
```

- The wedge is caught at **peak 13.7 N — well below the ~22 N break force** (force authority held; no breakage).
- The recovery is chosen **from the force signature alone** (no vision) and applied **within the same `F_max`**.
- The re-approach **seats** the bottle: base centered, at the cell floor.

The same loop is validated on a CPU fake env in `tests/test_recovery_loop.py` (jam → signature →
`rotate_align` → recover → success; budget never relaxed; unrecoverable jam escalates to `abort`).

## 6. Run it

```bash
export HOME=/workspace/persist/ovhome MPLBACKEND=Agg DISPLAY=:99 PYTHONPATH=/workspace/FORGE-plus_task3
# closed-loop recovery on a wedged insertion:
/workspace/.venv/bin/python scripts/run_recovery_insertion.py --jam 0.05 --backend heuristic
# clean insertion (no jam, no recovery):
/workspace/.venv/bin/python scripts/run_recovery_insertion.py --jam 0.0
# real local LLM instead of the heuristic:  --backend local   (Ollama)
```

Backends: `heuristic` (deterministic, force-reasoned, no API), `local` (Ollama), `anthropic`
(API key). All read only the text force signature — the recovery never sees an image.
