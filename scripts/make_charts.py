"""Render slide-ready PNG charts from results/results.json.

Needs matplotlib (a development convenience only; the package itself is stdlib-only):
    pip install matplotlib && python3 scripts/make_charts.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "charts")
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(RES, "results.json"), encoding="utf-8") as f:
    res = json.load(f)

INK, MUTE, GRID = "#1b2330", "#667085", "#e3e7ec"
BLUE, GREEN, RED, AMBER, GRAY = "#3b5bdb", "#1f9d63", "#cf3b33", "#b57900", "#98a2b3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "figure.dpi": 200,
})
S, G = "H1k3", "H2k3"
pct = lambda x: 100 * x[0]
err = lambda x: 100 * x[1]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. headline -------------------------------------------------------------
h = res["headline"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
labels = ["A\nno idle loop", "B\nnoise, no selector", "C\nfull Still Air"]
cols = [GRAY, AMBER, BLUE]
w = 0.36
for i, m in enumerate("ABC"):
    for j, (k, a) in enumerate(((S, 1.0), (G, 0.55))):
        ax[0].bar(i + (j - 0.5) * w, pct(h[m][k + ".acc"]), w, yerr=err(h[m][k + ".acc"]), color=cols[i], alpha=a,
                  capsize=3, error_kw={"ecolor": INK, "lw": 1})
        ax[0].text(i + (j - 0.5) * w, pct(h[m][k + ".acc"]) + 4, f"{pct(h[m][k + '.acc']):.0f}", ha="center", fontsize=11)
ax[0].set_xticks(range(3), labels)
ax[0].set_ylabel("two-hop accuracy (%)")
ax[0].set_ylim(0, 112)
ax[0].set_title("Accuracy: solid = 1 retrieval round, faded = 2", loc="left", fontsize=12, color=MUTE)
for i, m in enumerate("BC"):
    v = pct(h[m]["precision"])
    ax[1].bar(i, v, 0.55, yerr=err(h[m]["precision"]), color=cols[i + 1], capsize=3, error_kw={"ecolor": INK, "lw": 1})
    ax[1].text(i, v + 4, f"{v:.0f}%", ha="center")
ax[1].set_xticks(range(2), ["B\nno selector", "C\nselector"])
ax[1].set_ylim(0, 112)
ax[1].set_title("Memory precision (stored links that are true)", loc="left", fontsize=12, color=MUTE)
save(fig, "1_headline.png")

# 2. idle-budget sweep -----------------------------------------------------
rows = res["budget_sweep"]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
for mode, col, lab in (("noise_only", AMBER, "B  noise, no selector"), ("full", BLUE, "C  full Still Air")):
    rr = [r for r in rows if r["mode"] == mode]
    xs = [r["budget"] for r in rr]
    for a, k, t in ((ax[0], S, "1 retrieval round (strict)"), (ax[1], G, "2 retrieval rounds (generous)")):
        ys = [pct(r["summary"][k + ".acc"]) for r in rr]
        es = [err(r["summary"][k + ".acc"]) for r in rr]
        a.errorbar(xs, ys, yerr=es, color=col, marker="o", lw=2, capsize=3, label=lab)
        a.set_title(t, loc="left", fontsize=12, color=MUTE)
ax[0].set_ylabel("two-hop accuracy (%)")
ax[0].set_ylim(0, 105)
ax[1].set_ylim(60, 105)
ax[1].axhline(100, color=GRAY, ls="--", lw=1.2)
ax[1].text(xs[-1], 100.8, "A (no idle loop) = 100%", ha="right", color=MUTE, fontsize=11)
for a in ax:
    a.set_xlabel("idle tokens per session")
ax[0].legend(frameon=False, loc="upper left")
save(fig, "2_budget_sweep.png")

# 3. spontaneity vs constraint ------------------------------------------------
fr = [r for r in res["focus"] if r["use_state"]]
blind = next(r for r in res["focus"] if not r["use_state"])
fig, ax = plt.subplots(figsize=(7.6, 4.4))
xs = [r["amplitude"] for r in fr]
ax.errorbar(xs, [pct(r["asked_gain"]) for r in fr], yerr=[err(r["asked_gain"]) for r in fr], color=BLUE, marker="o", lw=2.4,
            capsize=3, label="questions it was asked (targeted)")
ax.errorbar(xs, [pct(r["unasked_gain"]) for r in fr], yerr=[err(r["unasked_gain"]) for r in fr], color=GREEN, marker="s", lw=2.4,
            capsize=3, label="questions never asked (broad)")
ax.axhline(pct(blind["asked_gain"]), color=BLUE, ls=":", lw=1.4)
ax.text(0.02, 13.5, "dotted line: blind noise", ha="left", color=BLUE, fontsize=10)
ax.set_xlabel("perturbation amplitude  (0 = pure internal drive, 1 = pure noise)")
ax.set_ylabel("accuracy gained (points)")
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1)
save(fig, "3_spontaneity_constraint.png")

# 4. drift failure vs amplitude -------------------------------------------------
am = res["amplitude_sweep"]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
xs = [r["amplitude"] for r in am]
ax.plot(xs, [pct(r["summary"]["repetition_rate"]) for r in am], color=RED, marker="o", lw=2.4, label="repetition (rumination)")
ax.plot(xs, [pct(r["summary"]["coverage"]) for r in am], color=GREEN, marker="s", lw=2.4, label="chains connected")
ax.set_xlabel("perturbation amplitude")
ax.set_ylabel("% of idle steps / % of chains")
ax.legend(frameon=False)
save(fig, "4_drift_vs_amplitude.png")

# 5. robustness heatmap ------------------------------------------------------------
rb = res["robustness"]
hs = sorted({r["h"] for r in rb})
ms = sorted({r["m"] for r in rb})
grid = [[next(pct(r["C"][G + ".acc"]) for r in rb if r["h"] == hh and r["m"] == mm) for mm in ms] for hh in hs]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
im = ax.imshow(grid, cmap="RdYlGn", vmin=65, vmax=118, aspect="auto", origin="lower")
ax.set_xticks(range(len(ms)), [f"{x:.2f}" for x in ms])
ax.set_yticks(range(len(hs)), [f"{x:.2f}" for x in hs])
ax.set_xlabel("grounding judge miss rate")
ax.set_ylabel("hallucination rate")
ax.grid(False)
for i in range(len(hs)):
    for j in range(len(ms)):
        ax.text(j, i, f"{grid[i][j]:.0f}", ha="center", va="center", color=INK, fontsize=12)
ax.set_title("C accuracy (%) with 2 rounds; A = 100", loc="left", fontsize=12, color=MUTE)
save(fig, "5_robustness.png")
print("charts written to", OUT, sorted(os.listdir(OUT)))
