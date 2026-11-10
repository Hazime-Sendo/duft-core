"""
Paper-scale Lyapunov measurement for Section 5 (32x32, T=10, dt=0.01).

For one background amplitude it evolves two trajectories that differ by a
1e-8 su(3) perturbation (Gell-Mann lambda_3 on link Ux(0,0)) and records,
every 10 steps:

  link_hi, link_lo : distances of the Fourier-transformed LINK variables U
                     (high-k: |k|>=k_cut=8, low-k: |k|<4).  NOT gauge
                     invariant.
  plaq_tot, plaq_hi, plaq_lo : the same, for the Wilson-plaquette field
                     (gauge invariant).
  relE_pct         : |H - H0| / H0 in percent along trajectory A.

It writes results/lyapunov_amp<amp>.csv and prints exponents obtained from
log-linear fits with three different fit-start times (t0 = 1, 3, 6), so
that the sensitivity of each exponent to the fitting window is visible.

Usage:  python3 scripts/lyapunov_paper_scale.py --amp 0.3 [--seed 1]
        (15-45 min per run on one CPU core; default seed 42 writes
        results/lyapunov_amp0.3.csv, other seeds ..._seed<k>.csv;
        --random: all-link random perturbation -> ..._random[_seed<k>].csv;
        --shell low|mid|high: random perturbation confined to one |k| shell
        -> ..._shell<name>[_seed<k>].csv)
One trajectory pair per run: see the caveats in the paper.
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.linalg import expm
from src.gauge_su3 import generate_background, leapfrog_step, compute_energy
from src.gauge_su3_fast import leapfrog_step_fast, compute_energy_fast
from scripts.lyapunov_phase_diagram import highk_lowk_distance

N, DT, T_END, K_CUT, DEFAULT_SEED, LOG_EVERY = 32, 0.01, 10.0, 8, 42, 10


def plaquette_field(Ux, Uy):
    """Vectorized Wilson plaquette; identical to core plaquette(i, j)."""
    Uy_ip = np.roll(Uy, -1, axis=0)
    Ux_jp = np.roll(Ux, -1, axis=1)
    dag = lambda M: np.swapaxes(M.conj(), -1, -2)
    return Ux @ Uy_ip @ dag(Ux_jp) @ dag(Uy)


def random_perturbation(UxB, UyB, seed, eps_total=1e-8):
    """Apply a random su(3) perturbation to EVERY link.

    Each link gets exp(i * e_l * H_l) with H_l a random traceless Hermitian
    matrix normalised to the Frobenius norm of lambda_3 (sqrt 2) and
    e_l = eps_total / sqrt(number of links), so that the total perturbation
    norm equals that of the single-link lambda_3 perturbation of size
    eps_total (same initial distance; only the spatial structure differs).
    Uses its own RNG stream, so the background for a given seed is the same
    as in the localized runs.
    """
    from src.gauge_su3 import make_hermitian_traceless
    prng = np.random.default_rng([seed, 1])
    n_links = 2 * UxB.shape[0] * UxB.shape[1]
    e = eps_total / np.sqrt(n_links)
    for U in (UxB, UyB):
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                H = make_hermitian_traceless(prng.normal(size=(3, 3)) + 1j * prng.normal(size=(3, 3)))
                H *= np.sqrt(2.0) / np.linalg.norm(H)
                U[i, j] = expm(1j * e * H) @ U[i, j]


SHELLS = {"low": (0.0, 4.0), "mid": (4.0, 8.0), "high": (8.0, 1e9)}   # |k| ranges [lo, hi)


def shell_perturbation(UxB, UyB, seed, shell, eps_total=1e-8):
    """Random su(3) perturbation restricted to one Fourier shell of |k|.

    A white random traceless-Hermitian field X_l (one per link, Ux and Uy)
    is filtered in Fourier space with the even mask lo <= |k| < hi
    (SHELLS: low |k|<4, mid 4<=|k|<8, high |k|>=8 -- the same split as the
    distance diagnostics), then rescaled so that the total Frobenius norm
    sqrt(sum_l ||X_l||_F^2) equals that of the single-link lambda_3
    perturbation (eps_total*sqrt(2)); links become exp(i X_l) U_l.  Even mask
    => real-space kernel is real => X_l stays Hermitian and traceless.
    Own RNG stream default_rng([seed, 2]); background identical to the other
    modes for the same seed.
    """
    n = UxB.shape[0]
    lo, hi = SHELLS[shell]
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    mask = ((K >= lo) & (K < hi)).astype(float)
    prng = np.random.default_rng([seed, 2])
    M = prng.normal(size=(2, n, n, 3, 3)) + 1j * prng.normal(size=(2, n, n, 3, 3))
    H = 0.5 * (M + np.conj(np.swapaxes(M, -1, -2)))
    tr = np.trace(H, axis1=-2, axis2=-1)
    H = H - tr[..., None, None] * np.eye(3) / 3.0
    F = np.fft.fft2(H, axes=(1, 2)) * mask[None, :, :, None, None]
    X = np.fft.ifft2(F, axes=(1, 2))
    X = 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))
    X *= eps_total * np.sqrt(2.0) / np.sqrt(np.sum(np.abs(X) ** 2))
    for d, U in enumerate((UxB, UyB)):
        for i in range(n):
            for j in range(n):
                U[i, j] = expm(1j * X[d, i, j]) @ U[i, j]


SPECTRA = ("lowpeak", "midpeak", "highpeak", "flat")
BACKGROUNDS = ("white", "smooth", "lowstrong", "highstrong")
SMOOTH_KMAX = 2.0     # smooth background: link field low-pass |k| <= 2
LOW_BOOST = 3.0       # lowstrong background: modes |k|<4 amplified by this factor (power x9)


def make_background(n, amp, rng, kind="white"):
    """white: the original random background (src.gauge_su3.generate_background).
    smooth: the SAME random su(3) field (same RNG draws), low-pass filtered to
        |k|<=SMOOTH_KMAX and rescaled to the same rms per link, then
        U = exp(i amp H): a nearly flat, slowly varying configuration (E=0).
    lowstrong: the same field with its |k|<4 modes multiplied by LOW_BOOST
        (long-wavelength structure on top of the white part).
    highstrong: the same field with its |k|>=8 modes multiplied by LOW_BOOST
        (short-wavelength structure emphasised)."""
    if kind == "white":
        return generate_background(n, n, amp, rng)
    from src.gauge_su3 import make_hermitian_traceless
    Hx = np.zeros((n, n, 3, 3), dtype=complex)
    Hy = np.zeros((n, n, 3, 3), dtype=complex)
    for i in range(n):
        for j in range(n):
            hx = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            hy = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            Hx[i, j] = make_hermitian_traceless(hx)
            Hy[i, j] = make_hermitian_traceless(hy)
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)

    def filt(H, mask):
        return np.fft.ifft2(np.fft.fft2(H, axes=(0, 1)) * mask[:, :, None, None], axes=(0, 1))

    out = []
    for H in (Hx, Hy):
        if kind == "smooth":
            Hn = filt(H, (K <= SMOOTH_KMAX).astype(float))
            Hn *= np.sqrt(np.sum(np.abs(H) ** 2) / np.sum(np.abs(Hn) ** 2))
        elif kind == "lowstrong":
            Hn = H + (LOW_BOOST - 1.0) * filt(H, (K < 4).astype(float))
        elif kind == "highstrong":
            Hn = H + (LOW_BOOST - 1.0) * filt(H, (K >= 8).astype(float))
        else:
            raise ValueError(kind)
        Hn = 0.5 * (Hn + np.conj(np.swapaxes(Hn, -1, -2)))
        out.append(Hn)
    Ux = np.zeros((n, n, 3, 3), dtype=complex)
    Uy = np.zeros((n, n, 3, 3), dtype=complex)
    for i in range(n):
        for j in range(n):
            Ux[i, j] = expm(1j * amp * out[0][i, j])
            Uy[i, j] = expm(1j * amp * out[1][i, j])
    Z = np.zeros_like(Ux)
    return Ux, Uy, Z.copy(), Z.copy()


def spectrum_weights(name, n):
    """Even, real Fourier weights w(k) for the link-perturbation field.
    lowpeak |k|<2 ; midpeak 5<=|k|<7 ; highpeak |k|>=12 ;
    flat: w = 1/c(k) with c(k)=2 sqrt(sin^2(pi kx/n)+sin^2(pi ky/n)) (lattice
    curl factor), w(0)=0, i.e. approximately white PLAQUETTE perturbation."""
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    if name == "lowpeak":
        return (K < 2).astype(float)
    if name == "midpeak":
        return ((K >= 5) & (K < 7)).astype(float)
    if name == "highpeak":
        return (K >= 12).astype(float)
    if name == "flat":
        c = 2 * np.sqrt(np.sin(np.pi * KX / n) ** 2 + np.sin(np.pi * KY / n) ** 2)
        w = np.zeros_like(c)
        nz = c > 0
        w[nz] = 1.0 / c[nz]
        return w
    raise ValueError(name)


def spectral_perturbation(UxB, UyB, seed, name, eps_total=1e-8):
    """Like shell_perturbation, but with the weights of `spectrum_weights`;
    own RNG stream default_rng([seed, 3]); same total link norm."""
    n = UxB.shape[0]
    wgt = spectrum_weights(name, n)
    prng = np.random.default_rng([seed, 3])
    M = prng.normal(size=(2, n, n, 3, 3)) + 1j * prng.normal(size=(2, n, n, 3, 3))
    H = 0.5 * (M + np.conj(np.swapaxes(M, -1, -2)))
    tr = np.trace(H, axis1=-2, axis2=-1)
    H = H - tr[..., None, None] * np.eye(3) / 3.0
    X = np.fft.ifft2(np.fft.fft2(H, axes=(1, 2)) * wgt[None, :, :, None, None], axes=(1, 2))
    X = 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))
    X *= eps_total * np.sqrt(2.0) / np.sqrt(np.sum(np.abs(X) ** 2))
    for d, U in enumerate((UxB, UyB)):
        for i in range(n):
            for j in range(n):
                U[i, j] = expm(1j * X[d, i, j]) @ U[i, j]


SPEC_EVERY = 100          # steps between recorded shell spectra (1 time unit)


def run(amp, seed=DEFAULT_SEED, random=False, shell=None, spectrum=None,
        n=N, t_end=T_END, engine="fast", return_spectra=False, bg="white"):
    step_fn = leapfrog_step_fast if engine == "fast" else leapfrog_step
    energy_fn = compute_energy_fast if engine == "fast" else compute_energy
    rng = np.random.default_rng(seed)
    UxA, UyA, ExA, EyA = make_background(n, amp, rng, bg)
    UxB, UyB, ExB, EyB = UxA.copy(), UyA.copy(), ExA.copy(), EyA.copy()
    if spectrum:
        spectral_perturbation(UxB, UyB, seed, spectrum)
    elif shell:
        shell_perturbation(UxB, UyB, seed, shell)
    elif random:
        random_perturbation(UxB, UyB, seed)
    else:
        lam3 = np.diag([1, -1, 0]).astype(complex)
        UxB[0, 0] = expm(1j * 1e-8 * lam3) @ UxB[0, 0]
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    nb = int(np.ceil(K.max())) + 1
    bins = np.rint(K).astype(int)
    cnt = np.bincount(bins.ravel(), minlength=nb)
    mid = (K >= 4) & (K < K_CUT)
    counts = (int((K < 4).sum()), int(mid.sum()), int((K >= K_CUT).sum()))
    H0 = energy_fn(ExA, EyA, UxA, UyA)
    rows, spectra = [], []
    for step in range(int(round(t_end / DT)) + 1):
        if step % LOG_EVERY == 0:
            Dh, Dl = highk_lowk_distance(UxA, UyA, UxB, UyB, K_CUT)
            dP = plaquette_field(UxA, UyA) - plaquette_field(UxB, UyB)
            F = np.fft.fft2(dP, axes=(0, 1))
            rows.append((step * DT, Dh, Dl,
                         np.sqrt(np.sum(np.abs(dP) ** 2)),
                         np.sqrt(np.sum(np.abs(F[K >= K_CUT]) ** 2)),
                         np.sqrt(np.sum(np.abs(F[K < 4]) ** 2)),
                         100 * abs(energy_fn(ExA, EyA, UxA, UyA) - H0) / abs(H0),
                         np.sqrt(np.sum(np.abs(F[mid]) ** 2)), *counts))
            if step % SPEC_EVERY == 0:
                P2 = (np.abs(F) ** 2).sum(axis=(-1, -2))
                spectra.append([step * DT] + list(np.bincount(bins.ravel(), weights=P2.ravel(),
                                                              minlength=nb) / np.maximum(cnt, 1)))
        step_fn(UxA, UyA, ExA, EyA, DT)
        step_fn(UxB, UyB, ExB, EyB, DT)
    if return_spectra:
        return np.array(rows), np.array(spectra)
    return np.array(rows)


NAMES = ["link_hi", "link_lo", "plaq_tot", "plaq_hi", "plaq_lo"]


def report(amp, d):
    t = d[:, 0]
    print(f"amp={amp}: exponents from log-linear fits, fit start t0 = 1 / 3 / 6")
    for k, n in enumerate(NAMES, start=1):
        out = []
        for t0 in (1.0, 3.0, 6.0):
            m = t > t0
            sl = np.polyfit(t[m], np.log(d[m, k]), 1)[0]
            r2 = np.corrcoef(t[m], np.log(d[m, k]))[0, 1] ** 2
            out.append(f"{sl:+.4f} (R2={r2:.2f})")
        print(f"  {n:9s} " + "   ".join(out))
    print(f"  energy drift [%]: max over trajectory = {d[:, 6].max():.5f}, "
          f"at t=T_end = {d[-1, 6]:.5f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("amp_pos", nargs="?", type=float, help="amplitude (positional, legacy)")
    ap.add_argument("--amp", type=float)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--random", action="store_true",
                    help="random su(3) perturbation on every link (same total norm as the single-link one)")
    ap.add_argument("--spectrum", choices=list(SPECTRA),
                    help="random perturbation with an artificial link spectrum (lowpeak/midpeak/highpeak/flat)")
    ap.add_argument("--bg", choices=list(BACKGROUNDS), default="white",
                    help="background: white (default, as before), smooth, lowstrong")
    ap.add_argument("--n", type=int, default=N, help="lattice size n x n (default 32)")
    ap.add_argument("--T", type=float, default=T_END, help="final time (default 10)")
    ap.add_argument("--engine", choices=["fast", "ref"], default="fast",
                    help="fast: vectorised src/gauge_su3_fast.py (default); ref: reference loops")
    ap.add_argument("--shell", choices=list(SHELLS),
                    help="random perturbation restricted to one |k| shell (same total norm)")
    a = ap.parse_args()
    amp = a.amp if a.amp is not None else a.amp_pos
    if amp is None:
        ap.error("amplitude required")
    d, spec = run(amp, a.seed, a.random, a.shell, a.spectrum, a.n, a.T, a.engine, return_spectra=True, bg=a.bg)
    os.makedirs("results", exist_ok=True)
    mode_tag = (f"_sp{a.spectrum}" if a.spectrum else f"_shell{a.shell}" if a.shell
                else "_random" if a.random else "")
    tag = mode_tag + ("" if a.seed == DEFAULT_SEED else f"_seed{a.seed}")
    prefix = "lyapunov_" if (a.n == N and a.T == T_END) else f"lyapunov_N{a.n}_T{a.T:g}_"
    bgtag = "" if a.bg == "white" else f"_bg{a.bg}"
    path = f"results/{prefix}amp{amp}{bgtag}{tag}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + NAMES + ["relE_pct", "plaq_mid", "n_lo", "n_mid", "n_hi"])
        w.writerows(d.tolist())
    with open(f"results/spectrum_{prefix[len('lyapunov_'):]}amp{amp}{bgtag}{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"k{b}" for b in range(spec.shape[1] - 1)])
        w.writerows(spec.tolist())
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"seed={a.seed} N={a.n} T={a.T:g} engine={a.engine} background={a.bg} perturbation="
              f"{'spectrum-' + a.spectrum if a.spectrum else 'shell-' + a.shell if a.shell else 'random(all links)' if a.random else 'single link'}")
        print(f"initial plaq distance = {d[0, 3]:.4e}, plaq_lo/plaq_hi at t=0: "
              f"{d[0, 5]:.3e}/{d[0, 4]:.3e}; at t=T_end: {d[-1, 5]:.3e}/{d[-1, 4]:.3e}")
        report(amp, d)
    text = buf.getvalue()
    print(text, end="")
    with open(path[:-4] + ".log", "w") as f:
        f.write(text)

