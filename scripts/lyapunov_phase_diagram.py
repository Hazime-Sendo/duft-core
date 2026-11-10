"""
Reproduces the amplitude phase diagram of Section 5 (Observation A):
Lyapunov exponent lambda as a function of background amplitude `amp`,
for both high-k and low-k modes.

Two nearly identical trajectories (A, B) are evolved from the same
random background, differing only by a 1e-8 perturbation along a
traceless (su(3)) direction -- NOT a pure-phase (U(1)) direction,
which the traceless-projected dynamics is blind to; see
docs/VERIFICATION_LOG.md.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.linalg import expm
from src.gauge_su3 import generate_background, leapfrog_step

seed = 42


def highk_lowk_distance(UxA, UyA, UxB, UyB, k_cut):
    Nx, Ny = UxA.shape[0], UxA.shape[1]
    kx = np.fft.fftfreq(Nx) * Nx
    ky = np.fft.fftfreq(Ny) * Ny
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Kmag = np.sqrt(KX ** 2 + KY ** 2)
    mask_hi = Kmag >= k_cut
    mask_lo = Kmag < 4

    def fft(U):
        return np.fft.fft2(U, axes=(0, 1))

    FxA, FyA = fft(UxA), fft(UyA)
    FxB, FyB = fft(UxB), fft(UyB)

    def dist(mask):
        dA = np.concatenate([FxA[mask].reshape(-1), FyA[mask].reshape(-1)])
        dB = np.concatenate([FxB[mask].reshape(-1), FyB[mask].reshape(-1)])
        return np.sqrt(np.sum(np.abs(dA - dB) ** 2))

    return dist(mask_hi), dist(mask_lo)


def measure_lambda(amp, Nx=32, Ny=32, dt=0.01, T_end=10.0,
                    k_cut=8, log_every=10, fit_from=3.0):
    rng = np.random.default_rng(seed)
    UxA, UyA, ExA, EyA = generate_background(Nx, Ny, amp, rng)
    UxB, UyB, ExB, EyB = UxA.copy(), UyA.copy(), ExA.copy(), EyA.copy()

    lam3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=np.complex128)
    UxB[0, 0] = expm(1j * 1e-8 * lam3) @ UxB[0, 0]

    n_steps = int(T_end / dt)
    t_log, Dh_log, Dl_log = [], [], []
    for step in range(n_steps + 1):
        if step % log_every == 0:
            Dh, Dl = highk_lowk_distance(UxA, UyA, UxB, UyB, k_cut)
            t_log.append(step * dt)
            Dh_log.append(Dh)
            Dl_log.append(Dl)
        leapfrog_step(UxA, UyA, ExA, EyA, dt)
        leapfrog_step(UxB, UyB, ExB, EyB, dt)

    t_log, Dh_log, Dl_log = map(np.array, (t_log, Dh_log, Dl_log))
    mask = t_log > fit_from
    lam_high = np.polyfit(t_log[mask], np.log(Dh_log[mask]), 1)[0]
    lam_low = np.polyfit(t_log[mask], np.log(Dl_log[mask]), 1)[0]
    return lam_high, lam_low, t_log, Dh_log, Dl_log


if __name__ == "__main__":
    # NOTE: the full amp = [0.05, 0.1, 0.2, 0.3] x T_end=10 x 32x32 scan
    # used in the paper takes several minutes per amplitude value; a
    # smaller lattice/T_end is used here for a quick smoke test.
    for amp in [0.05, 0.3]:
        t0 = time.time()
        lam_h, lam_l, *_ = measure_lambda(amp, Nx=16, Ny=16, T_end=5.0)
        print(f"amp={amp:.2f}  lambda_high={lam_h:.4f}  "
              f"lambda_low={lam_l:.4f}  ({time.time()-t0:.1f}s)")
