# Session handoff: write the arXiv paper + project webpage

You are writing (1) an arXiv-ready research paper and (2) a project webpage
for **FORGE-plus** — LLM-guided failure recovery for contact-rich assembly
under per-object force ceilings. All experimental work is done; this session
is writing, figure-making, and packaging. Repo: `robot-team00/FORGE-plus`,
branch `task3` (PR into `main` open — merge state may vary; the branch is
the source of truth).

## Read in this order (everything you claim must trace to one of these)

1. `README.md` — framing, architecture, design invariants, baselines,
   metrics, related work (with arXiv links).
2. `docs/proposal.html` — the original research proposal (motivation, §7
   algorithm, §10 metrics).
3. `docs/task1_gear_robotiq.md` — **the main experimental write-up**: the
   full 2F-140 gear arc (plant, PPO-failure evidence, BC/DAgger + weight
   soup, learned release head, table-pick staging, physical table-regrasp
   recovery, all headline numbers, open items).
4. `docs/task1_jam_recovery.md` — recovery baseline tables for BOTH
   grippers, with the honest-caveats sections (step caps, timeout
   economics, envelope forces).
5. `docs/task1_baselines.md` — budget baselines (ours / oracle / fixed).
6. `docs/task3/README.md` + `docs/task3/01…08` — the bottle-placement arc
   (learned insertion + release on the Franka, recovery loop, rendering,
   the 2F-140 asset build).
7. `checkpoints/README.md` — checkpoint provenance, including the failed
   PPO lineage kept as evidence.
8. `docs/REPLICATION_PLAYBOOK.md` — if you need methodology detail.

## The story the paper tells

Two-layer system: a **frozen LLM** sets a per-object force ceiling from
identity (no images) and, on failure, picks a recovery from a fixed menu by
reading a **compact text force signature**; a fast RL/BC skill executes
under a **hard force clamp** that the LLM cannot raise. Key invariants
(state them; they are the non-circularity of the eval): `F_break` is hidden
from every learned/LLM component; `F_max` is immutable during recovery;
force authority lives in the fast loop.

Results to feature:
- **Gear insertion, Robotiq 2F-140** (the flagship): one checkpoint, both classes,
  clean 256/256 + 0 breaks; learned release 256/256; full table-pick flow
  64/64 at 5.4 N mean peak; recovery sweep ours 40% vs vision-proxy 28%,
  heuristic 0%, press-harder 0% *futile*, none 0% with 20% breaks.
- **Gear insertion, Franka**: clean 200/200; sweep ours 64% / press-harder 96%
  breaks — note how press-harder shows a *different face* on each gripper
  (destructive vs futile): that contrast is a finding.
- **Bottle placement, Franka**: learned insertion + learned release of a fragile
  bottle, recovery caught at 16.1 N vs ~23 N break.
- **Negative results are first-class**: PPO-only provably fails at 0.4 mm
  clearance (9 runs, exploration-noise argument); tiny-std PPO polish
  destroys a working policy (1/std² amplification); release-head designs
  1–3 and *why* they fail (post-open label poison, trunk discards
  seat-state); the OSC lateral-traverse curse. These make the paper.

## Honesty constraints (non-negotiable, they survive into the paper)

- **Simulation only.** No sim-to-real claim; no fracture modeling
  (breakage = hidden scalar threshold on peak contact force).
- Learned-vs-scripted split is exactly as documented: insertion + release
  timing are learned; pick/carry staging and recovery *primitives* are
  scripted (LLM does the *selection*); label it like the docs and videos do.
- Keep the caveats: 2F-140 vs Franka sweep tables use different step caps
  (1600 vs 700 — not directly comparable across tables); table-flow
  recovery is a 3/5 smoke, not a full cell; `uni_rel_tp` release gate is a
  64-episode smoke, not 256; recovery episodes press harder than clean
  ones (envelope exposure = the residual breaks).
- Every number in the paper must be copy-checkable against a doc in the
  repo. Do not round in a direction that flatters.

## Deliverables

1. **`paper/` directory**: arXiv-ready LaTeX (`main.tex`, `references.bib`,
   `figures/`). Suggested skeleton: Intro (who sets the ceiling / what do
   you do on failure) → Related work (table in README maps it) → System
   (two layers, invariants, signature format, menu) → Benchmark (tasks,
   objects, hidden F_break, metrics §10) → Skill learning (PPO failure →
   demos/DAgger/BC → soup; release head) → Experiments (gates, sweeps,
   both grippers, force economy) → Limitations → Conclusion. Related-work
   arXiv IDs are already in `README.md`.
2. **Webpage**: a static project page (e.g. `site/index.html`, deployable
   via GitHub Pages) with the architecture figure, embedded videos
   (`docs/videos/task3/gear_clean_robotiq.mp4`,
   `gear_recovery_robotiq.mp4`, `forge_recovery_franka.mp4`,
   `forge_release.mp4`), headline tables, and links to paper/code.
3. Figures: `docs/architecture.svg`, `docs/render_preview.png`, and video
   stills — extract with `ffmpeg -ss <t> -i <mp4> -frames:v 1 out.png`
   (good moments are logged in the docs: clean seat ~k2291, recovery seat
   ~k6840, the JAM card frames). Plot the sweep tables as grouped bars
   (success/breaks/timeouts per recovery baseline, one panel per gripper).

## Practical notes

- Paper/webpage work needs no GPU pod; the repo alone suffices. If you do
  need the pod (new renders), read `CLAUDE.md` first — hard-won rules.
- Git: commit explicit paths only (never `git add -A`); push with plain
  `git push origin <branch>`; creds are already in the remote. Work on a
  new branch off `task3` (e.g. `paper`) unless told otherwise.
- LaTeX may not be installed wherever you run — `pdflatex` check first;
  compiling is nice-to-have, arXiv-ready source is the deliverable.
- Videos are tracked in-repo under `docs/videos/task3/`; embed by relative
  path so the page works on GitHub Pages from the repo.
