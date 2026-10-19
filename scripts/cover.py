"""Cover artwork for the README: deep blue and deep red on white, serif type, no institutional marks.
The motif is the book's own subject: lognormal densities fanning out with horizon over a J-curve."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "common"))
import numpy as np
from scipy import stats
from bookstyle import plt, BLUE, RED, GREY, ROOT

fig = plt.figure(figsize=(9, 4.2))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set_xlim(0, 9); ax.set_ylim(0, 4.2)
ax.add_patch(plt.Rectangle((0, 0), 9, 4.2, color="white"))
ax.add_patch(plt.Rectangle((0, 3.95), 9, 0.25, color=BLUE))
ax.add_patch(plt.Rectangle((0, 3.88), 9, 0.04, color=RED))
ax.add_patch(plt.Rectangle((0, 0), 9, 0.12, color=BLUE))
x = np.linspace(0.02, 4.0, 400)
for i, h in enumerate([1, 3, 6, 10, 15]):
    pdf = stats.lognorm(s=0.18 * np.sqrt(h), scale=np.exp(0.06 * h)).pdf(x)
    ax.plot(5.0 + x, 0.55 + 1.6 * pdf / 2.3, color=BLUE, lw=1.1, alpha=1 - 0.13 * i)
sys.path.insert(0, str(ROOT / "code" / "ch01"))
import ch01lib as L
tt, c, d, nav = L.fund_example()
cum = np.cumsum(d - c)
from scipy.interpolate import PchipInterpolator
t = np.linspace(0, 9, 300); j = PchipInterpolator(tt, cum)(t)
ax.plot(5.0 + 4 * t / 9, 2.95 + 0.75 * j / np.abs(cum).max(), color=RED, lw=1.6)
ax.text(0.45, 2.75, "Quantitative Finance", fontsize=27, color=BLUE)
ax.text(0.45, 2.20, "for Alternative Investments", fontsize=27, color=BLUE)
ax.text(0.47, 1.65, "Mathematics, Models and Code", fontsize=15, color=GREY, style="italic")
ax.plot([0.47, 4.3], [1.42, 1.42], color=RED, lw=0.9)
ax.text(0.47, 0.95, "JOHN KHAN", fontsize=13, color=BLUE)
out = ROOT / "figures" / "cover.png"
fig.savefig(out, dpi=200, metadata={"Software": None}, bbox_inches=None)
print(out)
