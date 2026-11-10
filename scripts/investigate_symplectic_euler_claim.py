"""
*** RESULT: this script does NOT reproduce the manuscript's claim. ***

This is an investigation script, not a verified result. It checks the
manuscript's claim (Section 3.4 / 6.1) that a naive, non-time-symmetric "symplectic Euler"
step produces energy growth of order 1e15 within O(1) time units,
INDEPENDENT of dt.

Running all three candidate "naive" schemes below, using the actual
verified force formulas (not a toy model), on a chaotic background
(amp=0.3), scanning both dt (at fixed T_end) and T_end (at fixed dt):
none of the three shows growth independent of dt. Schemes (A) and (B)
(the two single-kick symplectic-Euler orderings) show small, BOUNDED
energy error that shrinks monotonically as dt -> 0, consistent with
standard symplectic-integrator backward-error-analysis theory (a
first-order symplectic method shadows a nearby Hamiltonian and does
not exhibit secular energy drift). Scheme (C) (genuinely non-symplectic
explicit Euler) shows ordinary first-order error that grows slowly with
T_end but still shrinks cleanly as dt -> 0 -- not a dt-independent
catastrophe.

This means the manuscript's specific numerical claim (10^15-fold,
dt-independent blowup) could not be independently reproduced with any
natural reading of "naive symplectic Euler" applied to the verified
SU(3) dynamics in src/gauge_su3.py. Two possibilities: (1) the
original exploratory script that produced this number used a different
(and possibly buggy, e.g. non-unitary or linearized) update rule than
the one implemented here, in which case the manuscript's wording should
be corrected to describe whatever that scheme actually was, or (2) the
claim should be removed/qualified before submission, since it cannot
currently be backed by a verified script in this repository. See the
corresponding entry in docs/VERIFICATION_LOG.md.

This uses the actual verified SU(3) force formulas from
src/gauge_su3.py (not a toy linear oscillator), run on a chaotic
background (amp=0.3) for a long time window, with THREE candidate
"naive" (non-symmetric) stepping schemes:

  (A) single-kick symplectic Euler: E_new = E_old - dt*F(U_old);
                                     U_new = exp(i dt E_new) U_old
      (update momentum from old field, then field from NEW momentum --
      this is genuinely symplectic for separable Hamiltonians and is
      expected to show BOUNDED energy error, not blow up)

  (B) the other single-kick order:  U_new = exp(i dt E_old) U_old;
                                     E_new = E_old - dt*F(U_new)
      (update field from old momentum, then momentum from NEW field --
      also genuinely symplectic, same expectation)

  (C) fully explicit (forward) Euler, both updates from the OLD state:
      E_new = E_old - dt*F(U_old); U_new = exp(i dt E_old) U_old
      (NOT symplectic -- this is the one most likely to be unstable)

For each scheme we scan dt over several orders of magnitude at FIXED
T_end, and separately scan T_end at fixed dt, to see whether any
scheme produces growth that fails to shrink as dt -> 0.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.linalg import expm
from src.gauge_su3 import (
    generate_background, force_x, force_y, compute_energy,
)

seed = 42


def step_A(Ux, Uy, Ex, Ey, dt):
    """Single-kick symplectic Euler: momentum first (old field), then
    field from the NEW momentum."""
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    Fx = np.empty_like(Ex)
    Fy = np.empty_like(Ey)
    for i in range(Nx):
        for j in range(Ny):
            Fx[i, j] = force_x(Ux, Uy, i, j)
            Fy[i, j] = force_y(Ux, Uy, i, j)
    Ex -= dt * Fx
    Ey -= dt * Fy
    for i in range(Nx):
        for j in range(Ny):
            Ux[i, j] = expm(1j * dt * Ex[i, j]) @ Ux[i, j]
            Uy[i, j] = expm(1j * dt * Ey[i, j]) @ Uy[i, j]


def step_B(Ux, Uy, Ex, Ey, dt):
    """Single-kick symplectic Euler, other order: field first (old
    momentum), then momentum from the NEW field."""
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    for i in range(Nx):
        for j in range(Ny):
            Ux[i, j] = expm(1j * dt * Ex[i, j]) @ Ux[i, j]
            Uy[i, j] = expm(1j * dt * Ey[i, j]) @ Uy[i, j]
    Fx = np.empty_like(Ex)
    Fy = np.empty_like(Ey)
    for i in range(Nx):
        for j in range(Ny):
            Fx[i, j] = force_x(Ux, Uy, i, j)
            Fy[i, j] = force_y(Ux, Uy, i, j)
    Ex -= dt * Fx
    Ey -= dt * Fy


def step_C(Ux, Uy, Ex, Ey, dt):
    """Fully explicit forward Euler: both updates from the OLD state."""
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    Fx = np.empty_like(Ex)
    Fy = np.empty_like(Ey)
    Ux_old = Ux.copy()
    Uy_old = Uy.copy()
    for i in range(Nx):
        for j in range(Ny):
            Fx[i, j] = force_x(Ux_old, Uy_old, i, j)
            Fy[i, j] = force_y(Ux_old, Uy_old, i, j)
    for i in range(Nx):
        for j in range(Ny):
            Ux[i, j] = expm(1j * dt * Ex[i, j]) @ Ux_old[i, j]
            Uy[i, j] = expm(1j * dt * Ey[i, j]) @ Uy_old[i, j]
    Ex -= dt * Fx
    Ey -= dt * Fy


def run(step_fn, Nx, Ny, amp, dt, T_end, log_every=10):
    rng = np.random.default_rng(seed)
    Ux, Uy, Ex, Ey = generate_background(Nx, Ny, amp, rng)
    H0 = compute_energy(Ex, Ey, Ux, Uy)
    n_steps = int(T_end / dt)
    H_max = H0
    for step in range(n_steps):
        step_fn(Ux, Uy, Ex, Ey, dt)
        if step % log_every == 0:
            H = compute_energy(Ex, Ey, Ux, Uy)
            if not np.isfinite(H) or abs(H) > 1e8:
                return H0, H, True  # blew up / NaN
            H_max = max(H_max, abs(H))
    H_final = compute_energy(Ex, Ey, Ux, Uy)
    return H0, H_final, False


if __name__ == "__main__":
    Nx, Ny = 8, 8
    amp = 0.3

    for name, fn in [("A (EU-kick, momentum-first)", step_A),
                      ("B (EU-kick, field-first)", step_B),
                      ("C (explicit forward Euler)", step_C)]:
        print(f"\n=== Scheme {name} ===")
        print("-- dt scan at fixed T_end=2.0 --")
        for dt in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            H0, H1, blew_up = run(fn, Nx, Ny, amp, dt, T_end=2.0)
            ratio = (H1 / H0) if H0 != 0 else float("nan")
            print(f"  dt={dt:7.4f}  H0={H0:10.4f}  H_end={H1:14.6g}  "
                  f"H_end/H0={ratio:12.4g}  blew_up={blew_up}")

        print("-- T_end scan at fixed dt=0.01 --")
        for T_end in [1.0, 2.0, 5.0, 10.0, 20.0]:
            H0, H1, blew_up = run(fn, Nx, Ny, amp, dt=0.01, T_end=T_end)
            ratio = (H1 / H0) if H0 != 0 else float("nan")
            print(f"  T_end={T_end:6.2f}  H0={H0:10.4f}  H_end={H1:14.6g}  "
                  f"H_end/H0={ratio:12.4g}  blew_up={blew_up}")
