"""
DUFT source module: SU(3) link-variable gauge sector.

This module implements the verified Hamiltonian lattice gauge dynamics
used throughout the DUFT paper (Sections 2, 5, 6):

  - Wilson plaquette / staple construction (with the correct,
    direction-dependent dagger convention verified against the
    Wilson action by finite-difference comparison; see Section 6)
  - The force F_i(x) = [U_i(x) S_i(x)]_{AH, traceless}
  - A time-symmetric (half-kick / drift / half-kick) leapfrog step
  - A 4th-order Yoshida composition of the above
  - The gauge energy density (electric + magnetic)
  - The covariant Gauss's-law residual

All formulas here were validated against independent finite-difference
and direct numerical checks; see docs/VERIFICATION_LOG.md for the full
history of sign/normalization errors that were caught and corrected
during development.

Convention note: for the x-direction link, only the "lower" staple is
dagger-conjugated; for the y-direction link, both staples are
conjugated. This asymmetry is a consequence of the plaquette
orientation convention (see `plaquette`) and was confirmed correct by
direct comparison against the numerical derivative of the Wilson
action (Section 6 of the paper).
"""

import numpy as np
from scipy.linalg import expm

__all__ = [
    "make_hermitian_traceless",
    "generate_background",
    "plaquette",
    "staple_x", "staple_y",
    "force_x", "force_y",
    "leapfrog_step",
    "yoshida4_step",
    "compute_energy",
    "covariant_gauss_residual",
]


def make_hermitian_traceless(H):
    """Project a 3x3 complex matrix onto the su(3) Lie algebra
    (Hermitian, traceless)."""
    Hh = (H + H.conj().T) / 2.0
    return Hh - (np.trace(Hh) / 3.0) * np.eye(3)


def generate_background(Nx, Ny, amp, rng):
    """Generate a random SU(3) link configuration with small amplitude
    `amp`, using group-exponential (not linearized) unitary links.
    E is initialized to zero (temporal-gauge, source-free vacuum)."""
    Ux = np.zeros((Nx, Ny, 3, 3), dtype=np.complex128)
    Uy = np.zeros((Nx, Ny, 3, 3), dtype=np.complex128)
    Ex = np.zeros((Nx, Ny, 3, 3), dtype=np.complex128)
    Ey = np.zeros((Nx, Ny, 3, 3), dtype=np.complex128)
    for i in range(Nx):
        for j in range(Ny):
            Hx = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            Hy = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
            Ux[i, j] = expm(1j * amp * make_hermitian_traceless(Hx))
            Uy[i, j] = expm(1j * amp * make_hermitian_traceless(Hy))
    return Ux, Uy, Ex, Ey


def plaquette(Ux, Uy, i, j):
    """Wilson plaquette U_x(i,j) U_y(i+1,j) U_x(i,j+1)^dag U_y(i,j)^dag."""
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    ip, jp = (i + 1) % Nx, (j + 1) % Ny
    return Ux[i, j] @ Uy[ip, j] @ Ux[i, jp].conj().T @ Uy[i, j].conj().T


def staple_x(Ux, Uy, i, j):
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    ip, jp, jm = (i + 1) % Nx, (j + 1) % Ny, (j - 1) % Ny
    up = Uy[ip, j] @ Ux[i, jp].conj().T @ Uy[i, j].conj().T
    dn = (Uy[i, jm].conj().T @ Ux[i, jm] @ Uy[ip, jm]).conj().T
    return up + dn


def staple_y(Ux, Uy, i, j):
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    im, ip, jp = (i - 1) % Nx, (i + 1) % Nx, (j + 1) % Ny
    rt = (Ux[i, j] @ Uy[ip, j] @ Ux[i, jp].conj().T).conj().T
    lf = (Ux[im, j].conj().T @ Uy[im, j] @ Ux[im, jp]).conj().T
    return rt + lf


def force_x(Ux, Uy, i, j):
    S = staple_x(Ux, Uy, i, j)
    Omega = Ux[i, j] @ S
    return make_hermitian_traceless((Omega - Omega.conj().T) / (2j))


def force_y(Ux, Uy, i, j):
    S = staple_y(Ux, Uy, i, j)
    Omega = Uy[i, j] @ S
    return make_hermitian_traceless((Omega - Omega.conj().T) / (2j))


def leapfrog_step(Ux, Uy, Ex, Ey, dt):
    """Time-symmetric half-kick / drift / half-kick step. Modifies
    Ux, Uy, Ex, Ey in place."""
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    for i in range(Nx):
        for j in range(Ny):
            Ex[i, j] -= 0.5 * dt * force_x(Ux, Uy, i, j)
            Ey[i, j] -= 0.5 * dt * force_y(Ux, Uy, i, j)
    for i in range(Nx):
        for j in range(Ny):
            Ux[i, j] = expm(1j * dt * Ex[i, j]) @ Ux[i, j]
            Uy[i, j] = expm(1j * dt * Ey[i, j]) @ Uy[i, j]
    for i in range(Nx):
        for j in range(Ny):
            Ex[i, j] -= 0.5 * dt * force_x(Ux, Uy, i, j)
            Ey[i, j] -= 0.5 * dt * force_y(Ux, Uy, i, j)


_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W2 = -2.0 ** (1.0 / 3.0) / (2.0 - 2.0 ** (1.0 / 3.0))


def yoshida4_step(Ux, Uy, Ex, Ey, dt):
    """4th-order Yoshida composition of three leapfrog sub-steps
    (w1, w2, w1), w1 + w2 + w1 = 1. See Yoshida (1990)."""
    leapfrog_step(Ux, Uy, Ex, Ey, _W1 * dt)
    leapfrog_step(Ux, Uy, Ex, Ey, _W2 * dt)
    leapfrog_step(Ux, Uy, Ex, Ey, _W1 * dt)


def compute_energy(Ex, Ey, Ux, Uy):
    """Total gauge energy H = H_E + H_B.

    IMPORTANT: the magnetic term must be Tr(I - U_plaq) = 3 - Re Tr(U_plaq),
    NOT (1 - (1/3) Re Tr(U_plaq)). The latter is a factor-of-3
    normalization error that was caught during development: it does
    not match the potential whose variation gives `force_x`/`force_y`
    above, and produces spurious O(1) energy drift that does not
    vanish as dt -> 0 (see docs/VERIFICATION_LOG.md)."""
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    H_E = 0.0
    H_B = 0.0
    for i in range(Nx):
        for j in range(Ny):
            H_E += 0.5 * (
                np.trace(Ex[i, j] @ Ex[i, j].conj().T).real
                + np.trace(Ey[i, j] @ Ey[i, j].conj().T).real
            )
            H_B += np.trace(np.eye(3) - plaquette(Ux, Uy, i, j)).real
    return H_E + H_B


def covariant_gauss_residual(Ex, Ey, Ux, Uy):
    """||G|| for the standard Kogut-Susskind Gauss's law,
        G(x) = sum_i [ E_i(x) - U_i(x-i)^dag E_i(x-i) U_i(x-i) ].

    This is the formula confirmed, by direct comparison against an
    alternative (incorrect) candidate, to be exactly conserved under
    the dynamics above -- see docs/VERIFICATION_LOG.md."""
    Nx, Ny = Ux.shape[0], Ux.shape[1]
    s = 0.0
    for i in range(Nx):
        for j in range(Ny):
            im, jm = (i - 1) % Nx, (j - 1) % Ny
            tx = Ex[i, j] - Ux[im, j].conj().T @ Ex[im, j] @ Ux[im, j]
            ty = Ey[i, j] - Uy[i, jm].conj().T @ Ey[i, jm] @ Uy[i, jm]
            G = tx + ty
            s += np.sum(np.abs(G) ** 2)
    return np.sqrt(s)
