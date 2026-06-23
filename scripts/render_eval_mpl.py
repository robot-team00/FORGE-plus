#!/usr/bin/env python3
"""Render a learned-policy eval video with the LLM F_max budget overlay.
Reads /workspace/_scratch/rollout_<gripper>.npz (real policy rollouts) and
produces docs/eval_episode.mp4: per-episode side-view of the place action +
a live contact-force gauge vs the LLM's F_max budget and the hidden F_break.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import imageio.v2 as imageio

import sys
ROOT = "/workspace/FORGE-plus_task3"
RUN = sys.argv[1] if len(sys.argv) > 1 else "2026-06-21_forge-baseline"
GRIPPERS = [("franka_panda", "Franka Panda"), ("robotiq_2f140", "Robotiq 2F-140")]
RACK_TOP = 0.60
FPS = 20

def pick(npz, n_each=2):
    fb = npz["fbreak"]; rs = npz["reason"]; ds = npz["done_step"]
    sel = []
    # fragile-success, fragile-break, robust-success
    for cond, tag in [((rs==1)&(fb<45), "fragile - PLACED"),
                      ((rs==2)&(fb<45), "fragile - BROKEN"),
                      ((rs==1)&(fb>100), "robust - PLACED")]:
        idx = np.where(cond)[0]
        if len(idx):
            # prefer clear (later) terminations
            idx = idx[np.argsort(-ds[idx])]
            sel.append((int(idx[0]), tag))
    return sel

def draw_frame(g_label, eez, cf, fmax, fbreak, reason, dstep, t, Tend):
    fig = plt.figure(figsize=(12.8, 6.0), dpi=100)
    fig.patch.set_facecolor("#0e1117")
    # ---- left: scene side view ----
    ax = fig.add_axes([0.04, 0.10, 0.40, 0.78]); ax.set_facecolor("#0e1117")
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(0.40, 0.78); ax.axis("off")
    # table + rack
    ax.add_patch(Rectangle((-0.5, 0.40), 1.0, 0.06, color="#3a2f25"))
    ax.add_patch(Rectangle((-0.18, 0.55), 0.36, RACK_TOP-0.55, color="#6b5640"))
    ax.text(0, 0.565, "rack", color="#d8c7ad", ha="center", fontsize=9)
    z = eez[t]; ftip = z - 0.03
    over = cf[t] > fmax
    broke_now = (reason==2) and (t >= dstep >= 0)
    col = "#ff4d4d" if broke_now else ("#ffb000" if over else "#39d98a")
    # gripper (two fingers) + wrist
    ax.add_patch(Rectangle((-0.06, z+0.02), 0.12, 0.05, color="#9aa0a6"))
    ax.add_patch(Rectangle((-0.05, ftip), 0.018, z+0.02-ftip, color="#9aa0a6"))
    ax.add_patch(Rectangle((0.032, ftip), 0.018, z+0.02-ftip, color="#9aa0a6"))
    if cf[t] > 0.3:
        ax.scatter([0],[RACK_TOP+0.005], s=120+min(cf[t],fbreak)*6, c=col, alpha=0.5, zorder=5)
    ax.text(0, 0.75, g_label, color="#e6e6e6", ha="center", fontsize=12, weight="bold")
    # ---- right: force gauge over time ----
    ax2 = fig.add_axes([0.52, 0.14, 0.44, 0.72]); ax2.set_facecolor("#161b22")
    tt = np.arange(t+1)
    ymax = max(fmax*1.6, float(np.max(cf[:Tend]))*1.1, fbreak*1.15, 30)
    ax2.set_xlim(0, Tend); ax2.set_ylim(0, ymax)
    ax2.plot(tt, cf[:t+1], color=col, lw=2.4, zorder=4)
    ax2.axhline(fmax, color="#39d98a", ls="--", lw=1.8)
    ax2.axhline(fbreak, color="#ff4d4d", ls=":", lw=1.6)
    ax2.text(Tend*0.02, fmax+ymax*0.02, "LLM F_max budget = %.0f N" % fmax, color="#39d98a", fontsize=10)
    ax2.text(Tend*0.02, fbreak+ymax*0.02, "hidden F_break = %.0f N" % fbreak, color="#ff8080", fontsize=10)
    ax2.set_xlabel("control step", color="#aaa"); ax2.set_ylabel("contact force (N)", color="#aaa")
    ax2.tick_params(colors="#888")
    for sp in ax2.spines.values(): sp.set_color("#444")
    ax2.text(0.98, 0.95, "force: %.1f N" % cf[t], transform=ax2.transAxes, ha="right",
             color=col, fontsize=13, weight="bold")
    # outcome banner
    if reason==1 and t>=dstep>=0:
        fig.text(0.5, 0.94, "PLACED  (gentle contact within budget)", color="#39d98a", ha="center", fontsize=15, weight="bold")
    elif reason==2 and t>=dstep>=0:
        fig.text(0.5, 0.94, "BROKEN  (contact force exceeded fragility)", color="#ff4d4d", ha="center", fontsize=15, weight="bold")
    else:
        fig.text(0.5, 0.94, "placing...", color="#cccccc", ha="center", fontsize=14)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1]+(4,))
    plt.close(fig)
    return buf[:,:,:3].copy()

frames = []
for gid, glabel in GRIPPERS:
    fp = "%s/runs/%s/rollouts/%s.npz" % (ROOT, RUN, gid)
    if not os.path.exists(fp):
        print("skip (no rollout):", gid); continue
    d = np.load(fp)
    for ei, tag in pick(d):
        eez = d["eez"][:, ei]; cf = d["cf"][:, ei]
        fmax = float(d["fcmd"][ei]); fbreak = float(d["fbreak"][ei])
        ds = int(d["done_step"][ei]); rs = int(d["reason"][ei])
        Tend = (ds + 12) if ds > 0 else 120
        Tend = min(Tend, eez.shape[0]-1)
        title = "%s  -  %s   (F_max=%.0fN, F_break=%.0fN)" % (glabel, tag, fmax, fbreak)
        for t in range(0, Tend):
            fr = draw_frame(title, eez, cf, fmax, fbreak, rs, ds, t, Tend)
            frames.append(fr)
        # hold last frame
        for _ in range(int(FPS*1.2)):
            frames.append(frames[-1])


os.makedirs(os.path.join(ROOT, "runs", RUN, "videos"), exist_ok=True)
out = os.path.join(ROOT, "runs", RUN, "videos", "eval_episode.mp4")
imageio.mimsave(out, frames, fps=FPS, quality=8)
print("RENDER_DONE ->", out, "frames", len(frames), flush=True)
