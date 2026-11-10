"""
Reproduces the exact-conservation-of-Gauss's-law result of Section 6.

Initializes E to a nonzero random configuration (so the check is
nontrivial: G(0) != 0), then measures, over the trajectory and across
several dt values,
  (a) the maximum deviation of the norm, | ||G(t)|| - ||G(0)|| |, and
  (b) the maximum COMPONENT-WISE deviation, max over sites x and matrix
      elements (a,b) of |G_ab(x,t) - G_ab(x,0)|.
(b) is the stronger statement: conservation of every component at every
site, not only of the total norm.

Expected output: the deviation stays at the level of floating-point
round-off (~1e-15) for every dt tested, with no systematic
dt-dependence -- in contrast to an O(dt^n) convergence curve.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.gauge_su3 import (
    generate_background, leapfrog_step, covariant_gauss_residual,
    covariant_gauss_field,
    make_hermitian_traceless,
)

seed = 42
amp = 0.2


def init_with_nonzero_E(Nx, Ny, seed, amp):
    rng = np.random.default_rng(seed)
    Ux, Uy, Ex, Ey = generate_background(Nx, Ny, amp, rng)
    for i in range(Nx):
        for j in range(Ny):
            Ex[i, j] = 0.05 * make_hermitian_traceless(
                rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
            Ey[i, j] = 0.05 * make_hermitian_traceless(
                rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    return Ux, Uy, Ex, Ey


def run(Nx, Ny, dt_list, T_end, log_every=10):
    results = {}
    for dt in dt_list:
        Ux, Uy, Ex, Ey = init_with_nonzero_E(Nx, Ny, seed, amp)
        Gf0 = covariant_gauss_field(Ex, Ey, Ux, Uy)
        G0 = np.sqrt(np.sum(np.abs(Gf0) ** 2))
        n_steps = int(T_end / dt)
        max_dev = 0.0
        max_comp = 0.0
        for step in range(n_steps):
            leapfrog_step(Ux, Uy, Ex, Ey, dt)
            if step % log_every == 0:
                Gf = covariant_gauss_field(Ex, Ey, Ux, Uy)
                G = np.sqrt(np.sum(np.abs(Gf) ** 2))
                max_dev = max(max_dev, abs(G - G0))
                max_comp = max(max_comp, np.abs(Gf - Gf0).max())
        results[dt] = {"G0": G0, "max_dev": max_dev, "max_comp": max_comp}
        print(f"  dt={dt:.4f}  G0={G0:.6f}  max|norm drift|={max_dev:.3e}"
              f"  max|component drift|={max_comp:.3e}")
    return results


if __name__ == "__main__":
    dt_list = [0.02, 0.01, 0.005]
    for Nx in [6, 12, 32, 64]:
        print(f"=== {Nx}x{Nx} lattice ===")
        t0 = time.time()
        run(Nx, Nx, dt_list, T_end=2.0)
        print(f"  elapsed: {time.time()-t0:.1f}s\n")
