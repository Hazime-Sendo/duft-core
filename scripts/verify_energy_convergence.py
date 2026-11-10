"""
Reproduces the energy-convergence result of Section 3:
leapfrog is O(dt^2), 4th-order Yoshida is O(dt^4).

Expected output: the drift ratio upon halving dt converges to 4.00
(leapfrog) and 16.00 (Yoshida), matching the theoretical rates.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.gauge_su3 import (
    generate_background, leapfrog_step, yoshida4_step, compute_energy,
)

Nx, Ny = 8, 8
T_end = 6.0
seed = 42
amp = 0.2


def run(dt_list, log_every=5):
    results = {}
    for dt in dt_list:
        rng = np.random.default_rng(seed)
        Ux0, Uy0, Ex0, Ey0 = generate_background(Nx, Ny, amp, rng)
        H0 = compute_energy(Ex0, Ey0, Ux0, Uy0)
        n_steps = int(T_end / dt)

        Ux, Uy, Ex, Ey = Ux0.copy(), Uy0.copy(), Ex0.copy(), Ey0.copy()
        max_drift_lf = 0.0
        for step in range(n_steps):
            leapfrog_step(Ux, Uy, Ex, Ey, dt)
            if step % log_every == 0:
                max_drift_lf = max(max_drift_lf,
                                    abs(compute_energy(Ex, Ey, Ux, Uy) - H0))

        Ux, Uy, Ex, Ey = Ux0.copy(), Uy0.copy(), Ex0.copy(), Ey0.copy()
        max_drift_y4 = 0.0
        for step in range(n_steps):
            yoshida4_step(Ux, Uy, Ex, Ey, dt)
            if step % log_every == 0:
                max_drift_y4 = max(max_drift_y4,
                                    abs(compute_energy(Ex, Ey, Ux, Uy) - H0))

        results[dt] = {"lf": max_drift_lf, "y4": max_drift_y4}
        print(f"dt={dt:.4f}  lf_drift={max_drift_lf:.4e}  "
              f"y4_drift={max_drift_y4:.4e}")
    return results


if __name__ == "__main__":
    dt_list = [0.05, 0.02, 0.01, 0.005, 0.0025]
    t0 = time.time()
    results = run(dt_list)
    print(f"\nElapsed: {time.time()-t0:.1f}s\n")

    print("Convergence ratios (halving dt):")
    dts = sorted(dt_list, reverse=True)
    for k in range(len(dts) - 1):
        d1, d2 = dts[k], dts[k + 1]
        r_lf = results[d1]["lf"] / results[d2]["lf"]
        r_y4 = results[d1]["y4"] / results[d2]["y4"]
        print(f"  {d1} -> {d2}: leapfrog={r_lf:.2f} (theory 4), "
              f"yoshida4={r_y4:.2f} (theory 16)")
