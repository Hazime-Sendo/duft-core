"""
Vectorised (numpy-batched) version of the leapfrog step and energy in
src/gauge_su3.py.  Same formulas; the site loops are replaced by array
operations (np.roll for neighbours, batched matmul, batched eigh for the
matrix exponential of the Hermitian electric field).  The reference
implementation in gauge_su3.py remains the authoritative one;
tests/scripts compare the two (scripts/verify_fast_engine.py).

Why this is legitimate: within a half-kick the links are not changed, so the
force at every site depends only on the (fixed) links and the updates are
independent of the loop order; in the drift every link is updated
independently.
"""
import numpy as np

__all__ = ["leapfrog_step_fast", "compute_energy_fast", "plaquette_field_fast"]

_I3 = np.eye(3)


def _dag(M):
    return np.swapaxes(M.conj(), -1, -2)


def _roll(M, sx, sy):
    """M shifted so that result[i,j] = M[i+sx, j+sy] (periodic)."""
    return np.roll(M, (-sx, -sy), axis=(0, 1))


def _force(Ux, Uy):
    # staple_x
    up = _roll(Uy, 1, 0) @ _dag(_roll(Ux, 0, 1)) @ _dag(Uy)
    dn = _dag(_dag(_roll(Uy, 0, -1)) @ _roll(Ux, 0, -1) @ _roll(Uy, 1, -1))
    Om = Ux @ (up + dn)
    Fx = (Om - _dag(Om)) / 2j
    # staple_y
    rt = _dag(Ux @ _roll(Uy, 1, 0) @ _dag(_roll(Ux, 0, 1)))
    lf = _dag(_dag(_roll(Ux, -1, 0)) @ _roll(Uy, -1, 0) @ _roll(Ux, -1, 1))
    Om = Uy @ (rt + lf)
    Fy = (Om - _dag(Om)) / 2j

    def proj(H):
        Hh = (H + _dag(H)) / 2.0
        tr = np.trace(Hh, axis1=-2, axis2=-1)
        return Hh - tr[..., None, None] * _I3 / 3.0
    return proj(Fx), proj(Fy)


def _expi(E, dt):
    """exp(i dt E) for Hermitian E, batched."""
    w, V = np.linalg.eigh(E)
    return (V * np.exp(1j * dt * w)[..., None, :]) @ _dag(V)


def leapfrog_step_fast(Ux, Uy, Ex, Ey, dt):
    """Same contract as gauge_su3.leapfrog_step: modifies arrays in place."""
    Fx, Fy = _force(Ux, Uy)
    Ex -= 0.5 * dt * Fx
    Ey -= 0.5 * dt * Fy
    Ux[...] = _expi(Ex, dt) @ Ux
    Uy[...] = _expi(Ey, dt) @ Uy
    Fx, Fy = _force(Ux, Uy)
    Ex -= 0.5 * dt * Fx
    Ey -= 0.5 * dt * Fy


def plaquette_field_fast(Ux, Uy):
    return Ux @ _roll(Uy, 1, 0) @ _dag(_roll(Ux, 0, 1)) @ _dag(Uy)


def compute_energy_fast(Ex, Ey, Ux, Uy):
    HE = 0.5 * (np.sum(np.abs(Ex) ** 2) + np.sum(np.abs(Ey) ** 2))
    P = plaquette_field_fast(Ux, Uy)
    HB = (3.0 * Ux.shape[0] * Ux.shape[1] - np.trace(P, axis1=-2, axis2=-1).real.sum())
    return HE + HB
