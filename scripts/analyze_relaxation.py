"""
Test of the relaxation hypothesis for the low/high-k split of the plaquette
perturbation:  R(t) = plaq_lo / plaq_hi relaxes to an amplitude-dependent
common value, whatever the initial spectrum.

Uses the long-time / lattice-size / artificial-spectrum runs
(results/lyapunov_N<n>_T<T>_amp<amp>[_random|_shell<s>|_sp<name>]_seed<k>.csv,
written by lyapunov_paper_scale.py --n --T --spectrum --shell).

Quantities
  Rpm(t)  = [ (plaq_lo^2/n_lo) / (plaq_hi^2/n_hi) ]^(1/2)   per-mode amplitude
            ratio (lattice-size independent; the raw ratio plaq_lo/plaq_hi
            contains the trivial factor sqrt(n_hi/n_lo) ~ N).
  d ln R/dt = lambda_low - lambda_high  (exactly, for the log-linear fit).
  fit       ln Rpm(t) = A + B exp(-gamma t)   (A: asymptote, gamma: rate)
Outputs: a text summary on stdout and PNG figures in results/.
Usage: python3 scripts/analyze_relaxation.py [--outdir results]
"""
import os, re, sys, glob, csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(r"lyapunov_N(\d+)_T([\d.]+?)_amp([\d.]+?)(?:_bg(smooth|lowstrong))?(?:_(random|shell(?:low|mid|high)|sp[a-z]+))?_seed(\d+)\.csv$")
MODE_LABEL = {"random": "random (white)", "shelllow": "shell-low", "shellmid": "shell-mid",
              "shellhigh": "shell-high", "splowpeak": "lowpeak", "spmidpeak": "midpeak",
              "sphighpeak": "highpeak", "spflat": "flat (plaquette)"}
MODE_ORDER = ["splowpeak", "shelllow", "spmidpeak", "shellmid", "spflat", "random", "shellhigh", "sphighpeak"]


def load_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        d = np.array([[float(x) for x in row] for row in r])
    return {h: d[:, i] for i, h in enumerate(head)}


def collect(results, bg="white"):
    runs = {}
    for p in glob.glob(os.path.join(results, "lyapunov_N*_T*_amp*.csv")):
        m = PAT.search(os.path.basename(p))
        if not m:
            continue
        n, T, amp, mode, seed = int(m[1]), float(m[2]), m[3], m[5] or "single", int(m[6])
        if (m[4] or "white") != bg:
            continue
        d = load_csv(p)
        if "plaq_mid" not in d:
            continue
        rpm = np.sqrt((d["plaq_lo"] ** 2 / d["n_lo"]) / (d["plaq_hi"] ** 2 / d["n_hi"]))
        runs.setdefault((n, T, amp, mode), {})[seed] = dict(t=d["t"], rpm=rpm, d=d, path=p)
    return runs


def mean_lnR(group):
    seeds = sorted(group)
    L = np.array([np.log(group[s]["rpm"]) for s in seeds])
    return group[seeds[0]]["t"], L.mean(0), L.std(0, ddof=1) if len(seeds) > 1 else np.zeros(L.shape[1]), seeds


def relax_fit(t, y):
    f = lambda t, A, B, g: A + B * np.exp(-g * t)
    best = None
    for g0 in (0.05, 0.2, 0.6):
        try:
            p, _ = curve_fit(f, t, y, p0=(y[-1], y[0] - y[-1], g0), bounds=([-10, -10, 1e-3], [10, 10, 5]), maxfev=20000)
            r = np.sqrt(np.mean((f(t, *p) - y) ** 2))
            if best is None or r < best[1]:
                best = (p, r)
        except Exception:
            pass
    return best


def window_slope(t, y, lo, hi):
    m = (t >= lo) & (t <= hi)
    return np.polyfit(t[m], y[m], 1)[0]


def flatness(results, run):
    """std over |k|-bins 1..15 of ln(normalised per-mode power) at the final time, and the
    ratio of mean power in bins 1-3 to bins 12-15 (1 = flat)."""
    sp = os.path.join(results, "spectrum_" + os.path.basename(run["path"])[len("lyapunov_"):])
    if not os.path.exists(sp):
        return None
    row = np.genfromtxt(sp, delimiter=",", skip_header=1)[-1][1:]
    v = row[1:16] / row[1:16].sum()
    return np.std(np.log(v)), row[1:4].mean() / row[12:16].mean()


def bg_stats(bg, amp, n=32, seed=1):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import scripts.lyapunov_paper_scale as L
    from src.gauge_su3_fast import plaquette_field_fast
    Ux, Uy, _, _ = L.make_background(n, float(amp), np.random.default_rng(seed), bg)
    P = plaquette_field_fast(Ux, Uy)
    dev = np.mean(1 - np.trace(P, axis1=-2, axis2=-1).real / 3)
    F = np.fft.fft2(P - np.eye(3), axes=(0, 1))
    kx = np.fft.fftfreq(n) * n
    K = np.sqrt(np.add.outer(kx ** 2, kx ** 2))
    pw = (np.abs(F) ** 2).sum((-1, -2))
    lo, hi = (K < 4) & (K > 0), K >= 8      # k=0 (mean plaquette deviation) excluded
    return dev, np.sqrt(pw[lo].mean() / pw[hi].mean())


def main(results, outdir, bg="white"):
    runs = collect(results, bg)
    suffix = "" if bg == "white" else f"_{bg}"
    print(f"##### background: {bg} #####")
    for a in sorted({k[2] for k in runs}, key=float):
        dev, rr = bg_stats(bg, a)
        print(f"  background amp {a}: mean(1 - ReTrP/3) = {dev:.2e};  plaquette-field amplitude ratio low-k/high-k per mode = {rr:.2f}")
    if not runs:
        print("no runs found"); return
    keys = sorted(runs, key=lambda k: (k[0], k[1], float(k[2]), MODE_ORDER.index(k[3]) if k[3] in MODE_ORDER else 99))
    out = {}
    print("per-mode amplitude ratio Rpm = sqrt(mean power low / mean power high) of the plaquette difference;")
    print("values below are geometric means over seeds; lambda_low - lambda_high = d ln R / dt.\n")

    # ---------- (1) + (3): N=32 long time, all initial spectra
    for amp in sorted({k[2] for k in keys if k[0] == 32 and k[1] == 40}, key=float):
        sel = [k for k in keys if k[0] == 32 and k[1] == 40 and k[2] == amp]
        print(f"=== N=32, T=40, amp={amp}: Rpm(t) per initial spectrum ===")
        tt = [0, 2, 5, 10, 20, 30, 40]
        print("  initial spectrum   seeds   " + "  ".join(f"t={x:<4d}" for x in tt) + "   spread-of-ln(R) over seeds at T")
        fin = {}
        for k in sel:
            t, y, sd, seeds = mean_lnR(runs[k])
            idx = [int(np.argmin(abs(t - x))) for x in tt]
            print(f"  {MODE_LABEL.get(k[3], k[3]):17s} {len(seeds)}      " + "  ".join(f"{np.exp(y[i]):6.3f}" for i in idx)
                  + f"   {sd[-1]:.3f}")
            fin[k[3]] = (y, sd, t)
        for tcheck in (10, 20, 40):
            i = int(np.argmin(abs(fin[next(iter(fin))][2] - tcheck)))
            v = np.array([fin[m][0][i] for m in fin])
            print(f"  spread across initial spectra at t={tcheck}: max/min of Rpm = {np.exp(v.max() - v.min()):.3f} "
                  f"(ln-range {v.max() - v.min():.3f}; n spectra {len(v)})")
        print("  fit ln Rpm = A + B exp(-gamma t):  asymptote A -> R*=exp(A), rate gamma, rms residual")
        for k in sel:
            t, y, sd, seeds = mean_lnR(runs[k])
            fit = relax_fit(t, y)
            if fit:
                (A, B, g), r = fit
                note = "" if abs(B) > 0.15 else "  (starts near asymptote: gamma ill-determined)"
                print(f"    {MODE_LABEL.get(k[3], k[3]):17s} R*={np.exp(A):6.3f}  B={B:+.3f}  gamma={g:.3f}  rms={r:.3f}{note}")
                out[(32, amp, k[3])] = (A, B, g)
        print("  spectrum flatness at T (std of ln per-mode power over |k| bins 1-15; 0 = flat; low/high power ratio, 1 = flat):")
        for k in sel:
            fl = [flatness(results, runs[k][sd]) for sd in sorted(runs[k])]
            fl = np.array([f for f in fl if f])
            if len(fl):
                print(f"    {MODE_LABEL.get(k[3], k[3]):17s} std ln S = {fl[:, 0].mean():.3f}   P(bins1-3)/P(bins12-15) = {fl[:, 1].mean():.2f}")
        print("  lambda_low - lambda_high (= slope of ln Rpm) in consecutive windows [mean over seeds; seeds positive]:")
        wins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 40)]
        print("    " + " " * 18 + "  ".join(f"{a:>2d}-{b:<2d}      " for a, b in wins))
        for k in sel:
            cells = []
            for a, b in wins:
                sl = np.array([window_slope(runs[k][s]["t"], np.log(runs[k][s]["rpm"]), a, b) for s in sorted(runs[k])])
                cells.append(f"{sl.mean():+.3f}({int((sl > 0).sum())}/{len(sl)})")
            print(f"    {MODE_LABEL.get(k[3], k[3]):17s} " + "  ".join(f"{c:11s}" for c in cells))
        print()

    # amplitude dependence (random perturbation, N=32, T=40)
    amps = sorted({k[2] for k in keys if k[:2] == (32, 40.0) and k[3] == "random"}, key=float)
    if len(amps) > 1:
        print("=== amplitude dependence (random perturbation, N=32, T=40, seeds 1-3) ===")
        print("  amp    R0     R(10)   R(40)   asymptote R*   gamma")
        for a in amps:
            t, y, sd, _ = mean_lnR(runs[(32, 40.0, a, "random")])
            fit = relax_fit(t, y)
            i10 = int(np.argmin(abs(t - 10)))
            (A, B, g), r = fit
            print(f"  {a:5s} {np.exp(y[0]):6.3f} {np.exp(y[i10]):6.3f} {np.exp(y[-1]):6.3f}   {np.exp(A):6.3f}       {g:.3f}")
        print()

    # ---------- (2) lattice size
    for amp in sorted({k[2] for k in keys}, key=float):
        sel = [k for k in keys if k[2] == amp and k[3] == "random"]
        if len({k[0] for k in sel}) < 2:
            continue
        print(f"=== lattice-size dependence, amp={amp}, random perturbation (seeds 1-3) ===")
        tt = [0, 2, 5, 10, 20]
        print("  N      T    seeds  " + "  ".join(f"t={x:<4d}" for x in tt) + "   fit: R*     gamma   | slope(5-10) slope(10-20)")
        for k in sorted(sel):
            t, y, sd, seeds = mean_lnR(runs[k])
            idx = [int(np.argmin(abs(t - x))) for x in tt if x <= t[-1]]
            fit = relax_fit(t[t <= 20], y[t <= 20])
            (A, B, g), r = fit
            s1 = np.array([window_slope(runs[k][s]["t"], np.log(runs[k][s]["rpm"]), 5, 10) for s in seeds])
            s2 = np.array([window_slope(runs[k][s]["t"], np.log(runs[k][s]["rpm"]), 10, 20) for s in seeds])
            yl = y[t <= 20][-1]
            flag = "" if abs(A - yl) < 0.3 else "  [fit extrapolates beyond data: R*, gamma unreliable]"
            print(f"  {k[0]:4d}  {k[1]:4.0f}  {len(seeds)}      " + "  ".join(f"{np.exp(y[i]):6.3f}" for i in idx)
                  + f"        {np.exp(A):6.3f}  {g:6.3f}  | {s1.mean():+.3f}({int((s1 > 0).sum())}/{len(s1)}) {s2.mean():+.3f}({int((s2 > 0).sum())}/{len(s2)})" + flag)
        print()

    # ---------- figures
    os.makedirs(outdir, exist_ok=True)
    cols = {"splowpeak": "#4c1d95", "shelllow": "#7c3aed", "spmidpeak": "#0e7490", "shellmid": "#06b6d4",
            "spflat": "#15803d", "random": "#374151", "shellhigh": "#ea580c", "sphighpeak": "#b91c1c"}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, amp in zip(axs, ["0.1", "0.2"]):
        for k in keys:
            if k[:3] == (32, 40.0, amp):
                t, y, sd, seeds = mean_lnR(runs[k])
                ax.plot(t, np.exp(y), color=cols.get(k[3], "k"), lw=1.6, label=MODE_LABEL.get(k[3], k[3]))
                ax.fill_between(t, np.exp(y - sd), np.exp(y + sd), color=cols.get(k[3], "k"), alpha=0.12, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(f"amp = {amp}, 32x32, mean of seeds 1-3")
        ax.set_ylabel("Rpm = per-mode amplitude ratio, low-k / high-k"); ax.grid(alpha=0.25)
    axs[1].legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"relax{suffix}_fig1_ratio_vs_time.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, amp in zip(axs, ["0.1", "0.2"]):
        for k in sorted(keys):
            if k[2] == amp and k[3] == "random":
                t, y, sd, seeds = mean_lnR(runs[k])
                ax.plot(t, np.exp(y), lw=1.8, label=f"{k[0]}x{k[0]}")
                ax.fill_between(t, np.exp(y - sd), np.exp(y + sd), alpha=0.12, lw=0)
        ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_title(f"lattice size, amp = {amp}, random perturbation")
        ax.set_ylabel("Rpm"); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"relax{suffix}_fig2_lattice_size.png"), dpi=150); plt.close(fig)

    # shell-resolved spectra: initial and final, per initial spectrum
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for r_, amp in enumerate(["0.1", "0.2"]):
        for c_, which in enumerate(["t=0", "t=40"]):
            ax = axs[r_, c_]
            for k in keys:
                if k[:3] != (32, 40.0, amp):
                    continue
                accum = []
                for s in sorted(runs[k]):
                    sp = os.path.join(results, "spectrum_" + os.path.basename(runs[k][s]["path"])[len("lyapunov_"):])
                    if os.path.exists(sp):
                        dd = np.genfromtxt(sp, delimiter=",", skip_header=1)
                        row = dd[0] if which == "t=0" else dd[-1]
                        accum.append(np.log(row[1:] / row[1:17].sum()))   # normalise by sum over k-bins 0..15
                if accum:
                    kk = np.arange(len(accum[0]))
                    ax.plot(kk[:23], np.exp(np.mean(accum, 0))[:23], color=cols.get(k[3], "k"), lw=1.4,
                            label=MODE_LABEL.get(k[3], k[3]))
            ax.set_yscale("log"); ax.set_xlabel("|k| (bin)"); ax.grid(alpha=0.25)
            ax.set_ylabel("mean power per mode / sum(bins 0-15)"); ax.set_title(f"plaquette-difference spectrum, amp {amp}, {which}")
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"relax{suffix}_fig3_spectra.png"), dpi=150); plt.close(fig)

    # spectrum collapse metric
    print("=== shell-resolved spectrum collapse (N=32, T=40; normalised per-mode power, bins 1-15) ===")
    for amp in ["0.1", "0.2"]:
        for which, idx in (("t=0", 0), ("t=10", 10), ("t=40", -1)):
            vecs = {}
            for k in keys:
                if k[:3] != (32, 40.0, amp):
                    continue
                acc = []
                for s in sorted(runs[k]):
                    sp = os.path.join(results, "spectrum_" + os.path.basename(runs[k][s]["path"])[len("lyapunov_"):])
                    if os.path.exists(sp):
                        dd = np.genfromtxt(sp, delimiter=",", skip_header=1)
                        row = dd[idx]
                        acc.append(np.log(row[2:17] / row[2:17].sum()))
                if acc:
                    vecs[k[3]] = np.mean(acc, 0)
            if len(vecs) > 1:
                M = np.array(list(vecs.values()))
                dev = np.abs(M - M.mean(0)).max(1)
                print(f"  amp {amp} {which:5s}: max over k-bins of |ln S_a - mean ln S| = {np.max(dev):.2f} (worst: "
                      f"{MODE_LABEL.get(list(vecs)[int(np.argmax(dev))], '?')}); median over spectra {np.median(dev):.2f}; "
                      f"typical bin-to-bin variation of the mean spectrum {np.std(M.mean(0)):.2f}")


def compare_backgrounds(results):
    print("\n##### comparison of backgrounds (N=32, T=40, mean over seeds; Rpm at t=0 / 10 / 40 and final flatness) #####")
    all_runs = {bg: collect(results, bg) for bg in ("white", "smooth", "lowstrong")}
    amps = sorted({k[2] for r in all_runs.values() for k in r if k[:2] == (32, 40.0)}, key=float)
    for amp in amps:
        print(f"--- amp {amp} ---")
        print("  initial spectrum    " + "".join(f"| {bg:^36s}" for bg in all_runs))
        for mode in MODE_ORDER:
            cells = []
            for bg, r in all_runs.items():
                k = (32, 40.0, amp, mode)
                if k not in r:
                    cells.append(f"| {'-':^36s}")
                    continue
                t, y, sd, seeds = mean_lnR(r[k])
                i10 = int(np.argmin(abs(t - 10)))
                fl = [flatness(results, r[k][s]) for s in seeds]
                fl = np.array([f for f in fl if f])
                fs = f"{fl[:, 0].mean():.2f}" if len(fl) else "n/a"
                cells.append(f"| {np.exp(y[0]):5.2f} {np.exp(y[i10]):5.2f} {np.exp(y[-1]):5.2f} flat {fs:>5s} n={len(seeds)}   ")
            if any("-" != c.strip("| ").strip() for c in cells):
                print(f"  {MODE_LABEL.get(mode, mode):18s}" + "".join(cells))


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(__file__), "..", "results")
    outdir = sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else here
    bgs = [sys.argv[sys.argv.index("--bg") + 1]] if "--bg" in sys.argv else ["white", "smooth", "lowstrong"]
    for b in bgs:
        if collect(here, b):
            main(here, outdir, b)
    if len(bgs) > 1:
        compare_backgrounds(here)
