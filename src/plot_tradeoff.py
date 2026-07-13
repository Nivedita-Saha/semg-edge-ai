"""
Step 6.2 — The trade-off chart.

Accuracy against size, and accuracy against latency.

The Pareto frontier connects the models that are not beaten on BOTH axes
by any other model. A point below the frontier is strictly dominated:
some other model is smaller AND more accurate, so there is no reason to
ever choose it. Identifying that is the whole job of an edge-AI engineer.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

with open("results/benchmark.json") as f:
    results = json.load(f)

names = [r["name"] for r in results]
sizes = np.array([r["size_kb"] for r in results])
lats = np.array([r["latency_ms"] for r in results])
accs = np.array([r["accuracy"] for r in results])

# Colour by family
colours = {
    "Baseline": "#444444",
    "Baseline + int8": "#888888",
    "Pruned (48%)": "#1f77b4",
    "Pruned + int8": "#6baed6",
    "Student (distilled)": "#d62728",
    "Student + int8": "#ff9896",
}


def pareto_front(x, y):
    """
    Return indices on the Pareto frontier: minimise x, maximise y.
    A point is on the frontier if nothing else is both smaller AND better.
    """
    idx = np.argsort(x)
    front, best_y = [], -np.inf
    for i in idx:
        if y[i] > best_y:
            front.append(i)
            best_y = y[i]
    return front


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

for ax, xvals, xlabel, title in [
    (ax1, sizes, "Model size (KB)", "Accuracy vs Size"),
    (ax2, lats, "Inference latency (ms)", "Accuracy vs Latency"),
]:
    front = pareto_front(xvals, accs)
    fx = [xvals[i] for i in front]
    fy = [accs[i] for i in front]
    ax.plot(fx, fy, "--", color="grey", alpha=0.6, zorder=1,
            label="Pareto frontier")

    for i, name in enumerate(names):
        ax.scatter(xvals[i], accs[i], s=190, zorder=3,
                   color=colours[name], edgecolor="white", linewidth=1.5)
        ax.annotate(name, (xvals[i], accs[i]),
                    xytext=(8, 8), textcoords="offset points", fontsize=9)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(min(accs) - 2, max(accs) + 2)

fig.suptitle(
    "Model compression trade-offs — sEMG gesture recognition (Ninapro DB5, 17 classes)\n"
    "All models converted to TensorFlow Lite and benchmarked identically",
    fontsize=13, y=1.02,
)

plt.tight_layout()
plt.savefig("figures/tradeoff_chart.png", dpi=200, bbox_inches="tight")
print("Saved: figures/tradeoff_chart.png")

# --- Which models are on the frontier? ---
print()
print("=" * 66)
print("PARETO FRONTIER (accuracy vs size)")
print("=" * 66)
front = set(pareto_front(sizes, accs))
for i, name in enumerate(names):
    mark = "ON FRONTIER" if i in front else "  dominated"
    print(f"  {mark}   {name:<22} {sizes[i]:>6.1f} KB   {accs[i]:>6.2f}%")

print()
print("A dominated model is smaller-AND-worse than something else on the")
print("list. There is no scenario in which you would deploy it.")

plt.show()
