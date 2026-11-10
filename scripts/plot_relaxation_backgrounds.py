"""Rpm(t) for three initial spectra (lowpeak dashed, random solid, highpeak dotted) in the three
backgrounds (white, smooth, lowstrong), N=32, T=40, mean of seeds 1-3 (log scale)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import analyze_relaxation as A

here = os.path.join(os.path.dirname(__file__), "..", "results")
R = {bg: A.collect(here, bg) for bg in ("white", "smooth", "lowstrong")}
col = {"white": "#374151", "smooth": "#0e7490", "lowstrong": "#b45309"}
ls = {"splowpeak": "--", "random": "-", "sphighpeak": ":"}
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, amp in zip(axs, ["0.1", "0.2", "0.5"]):
    for bg, runs in R.items():
        for mode, style in ls.items():
            k = (32, 40.0, amp, mode)
            if k in runs:
                t, y, sd, _ = A.mean_lnR(runs[k])
                ax.plot(t, np.exp(y), color=col[bg], ls=style, lw=1.5, label=f"{bg}, {A.MODE_LABEL[mode]}")
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(f"amp = {amp}"); ax.grid(alpha=0.25)
    ax.set_ylabel("Rpm (low-k / high-k, per mode)")
axs[0].legend(fontsize=6.5, frameon=False, ncol=1)
fig.tight_layout(); fig.savefig(os.path.join(here, "relax_fig4_backgrounds.png"), dpi=150)
