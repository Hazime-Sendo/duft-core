"""
Window-sensitivity analysis of the finite-time exponents (Section 5).

Reads results/lyapunov_amp<amp>.csv written by lyapunov_paper_scale.py
(two-trajectory distances: link and Wilson-plaquette, low-k |k|<4 and
high-k |k|>=8) and fits log(distance) over several time windows.

Differences from a quick draft that compared one trajectory with its own
initial state, or took Fourier index [-3:] as 'high k':
  * distances are between TWO trajectories (perturbed vs reference);
  * the k-split uses |k| from fftfreq (index [-3:] is k = -3..-1, i.e. LOW k);
  * windows lie inside T=10; every fit reports R^2 and is skipped if it
    would need fewer than 5 points or non-positive distances.
"""
import sys, os, csv, glob
import numpy as np

WINDOWS = [(3, 7), (5, 9), (7, 10)]
FINE_WINDOWS = [(2, 5), (4, 7), (6, 9), (8, 10)]   # finer windows (--fine)
COLS = ["link_lo", "link_hi", "plaq_lo", "plaq_hi", "plaq_tot"]


def load(path):
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        d = np.array([[float(x) for x in row] for row in r])
    return {h: d[:, i] for i, h in enumerate(head)}


def lyapunov_slope(t, dist):
    """Log-linear fit; returns (slope, R^2) or (nan, nan) if unusable."""
    if len(t) < 5 or np.any(dist <= 0):
        return float("nan"), float("nan")
    y = np.log(dist)
    slope, icpt = np.polyfit(t, y, 1)
    ss_res = np.sum((y - (slope * t + icpt)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return slope, (1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def analyze(ts, windows=None):
    out = {}
    for lo, hi in (windows or WINDOWS):
        m = (ts["t"] >= lo) & (ts["t"] <= hi)
        out[(lo, hi)] = {c: lyapunov_slope(ts["t"][m], ts[c][m]) for c in COLS}
    return out


def parse_name(path):
    """lyapunov_amp0.1[_random|_shell<low|mid|high>][_seed<k>].csv -> (amp, mode, seed);
    mode is 'local', 'random' or 'shell-<name>'."""
    b = os.path.basename(path)[len("lyapunov_amp"):-len(".csv")]
    seed = 42
    if "_seed" in b:
        b, sd = b.split("_seed")
        seed = int(sd)
    mode = "local"
    if b.endswith("_random"):
        b, mode = b[:-len("_random")], "random"
    for sh in ("low", "mid", "high"):
        if b.endswith("_shell" + sh):
            b, mode = b[:-len("_shell" + sh)], "shell-" + sh
    return b, mode, seed


def amp_seed_of(path):          # backward-compatible helper
    amp, mode, seed = parse_name(path)
    return amp, seed


def print_one(path):
    amp, mode, seed = parse_name(path)
    print(f"amp={amp} perturbation={mode} seed={seed}")
    for w, res in analyze(load(path)).items():
        print(f"  window {w}: " + "  ".join(
            f"{c}={res[c][0]:+.3f}(R2={res[c][1]:.2f})" for c in COLS))


def _diff(res, w):
    r = res[w]
    return r["plaq_lo"][0] - r["plaq_hi"][0]


MODE_ORDER = ["local", "random", "shell-low", "shell-mid", "shell-high"]


def _groups(paths, windows):
    groups = {}
    for p in paths:
        amp, mode, seed = parse_name(p)
        groups.setdefault((amp, mode), {})[seed] = analyze(load(p), windows)
    return groups


def print_summary(paths, windows=None):
    """Per amplitude and perturbation type: plaquette low-k minus high-k
    exponent per seed/window, then paired comparisons (same seed, same
    background) of every other perturbation type against 'local' and
    'random', then the amplitude scan of the random perturbation."""
    windows = windows or WINDOWS
    groups = _groups(paths, windows)
    key = lambda k: (float(k[0]), MODE_ORDER.index(k[1]))
    for (amp, mode) in sorted(groups, key=key):
        seeds = sorted(groups[(amp, mode)])
        G = groups[(amp, mode)]
        print(f"\n=== amp={amp} perturbation={mode}: plaquette low-k minus high-k "
              f"exponent (positive = low-k grows faster); seeds {seeds} ===")
        for w in windows:
            diffs = np.array([_diff(G[sd], w) for sd in seeds])
            row = "  ".join(f"s{sd}:{G[sd][w]['plaq_lo'][0]:+.2f}/{G[sd][w]['plaq_hi'][0]:+.2f}"
                            for sd in seeds)
            sd_ = np.std(diffs, ddof=1) if len(diffs) > 1 else float("nan")
            print(f"  window {w}: {row}")
            print(f"     diff mean={np.mean(diffs):+.3f} sd={sd_:.3f} "
                  f"seeds with diff>0: {int(np.sum(diffs > 0))}/{len(diffs)}")
        print(f"  plaq_tot exponent (window {windows[0]}): " + "  ".join(
            f"s{sd}:{G[sd][windows[0]]['plaq_tot'][0]:+.3f}" for sd in seeds))
    # paired comparisons against local / random references
    for amp in sorted({a for a, _ in groups}, key=float):
        for ref in ("local", "random"):
            for mode in MODE_ORDER:
                if mode == ref or (amp, mode) not in groups or (amp, ref) not in groups:
                    continue
                if ref == "random" and mode == "local":
                    continue
                if ref == "local" and mode == "random" and False:
                    continue
                common = sorted(set(groups[(amp, mode)]) & set(groups[(amp, ref)]))
                if not common:
                    continue
                print(f"\n=== amp={amp}: paired {mode} - {ref}, seeds {common} "
                      f"(diff_{mode} - diff_{ref}; <0: {mode} has weaker low-k lead) ===")
                for w in windows:
                    da = np.array([_diff(groups[(amp, mode)][sd], w) for sd in common])
                    db = np.array([_diff(groups[(amp, ref)][sd], w) for sd in common])
                    dd = da - db
                    sd_ = np.std(dd, ddof=1) if len(dd) > 1 else float("nan")
                    print(f"  window {w}: {mode} {np.mean(da):+.3f}  {ref} {np.mean(db):+.3f}  "
                          f"paired diff {np.mean(dd):+.3f} (sd {sd_:.3f})")
    print_amp_scan(groups, windows)


def print_amp_scan(groups, windows, mode="random", seeds=(1, 2, 3)):
    amps = sorted({a for a, m in groups if m == mode and set(seeds) <= set(groups[(a, m)])}, key=float)
    if len(amps) < 2:
        return
    print(f"\n=== amplitude scan, perturbation={mode}, seeds {list(seeds)} (same seeds for every amplitude): "
          f"plaquette low-k minus high-k exponent, mean +/- sd (n positive / n) ===")
    print("  amp   " + "   ".join(f"{str(w):>22s}" for w in windows))
    for a in amps:
        cells = []
        for w in windows:
            d = np.array([_diff(groups[(a, mode)][sd], w) for sd in seeds])
            cells.append(f"{d.mean():+.3f}+/-{d.std(ddof=1):.3f} ({int((d > 0).sum())}/{len(d)})")
        print(f"  {a:5s} " + "   ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    summary = "--summary" in sys.argv
    fine = "--fine" in sys.argv
    windows = FINE_WINDOWS if fine else WINDOWS
    here = os.path.join(os.path.dirname(__file__), "..", "results")
    paths = args or sorted(glob.glob(os.path.join(here, "lyapunov_amp*.csv")))
    if summary:
        print_summary(paths, windows)
    else:
        for p in paths:
            amp, mode, seed = parse_name(p)
            print(f"amp={amp} perturbation={mode} seed={seed}")
            for w, res in analyze(load(p), windows).items():
                print(f"  window {w}: " + "  ".join(
                    f"{c}={res[c][0]:+.3f}(R2={res[c][1]:.2f})" for c in COLS))
