"""
Reproduces the adiabatic-elimination result of Section 2.3: for the scalar toy model with linear
restoring "force" F(A) = A and friction coefficient gamma,

    dA/dt = E
    dE/dt = -F(A) - gamma*E = -A - gamma*E,

the true damped trajectory at large gamma should converge onto the
adiabatic (slaved) prediction

    dA/dt ~= -F(A)/gamma = -A/gamma   =>   A(t) ~= A(0) * exp(-t/gamma),

and NOT onto the naive (dimensionally inconsistent) guess
dA/dt ~= -F(A) = -A, i.e. A(t) ~= A(0) * exp(-t).

This is solved as a genuinely coupled 2x2 linear ODE system (exact
eigenvalue solution -- not an approximation), so the comparison below
is between the EXACT solution of the full (A, E) system and the
adiabatic-elimination prediction, as a function of gamma.
"""
import numpy as np


def exact_A_of_t(A0, E0, gamma, t):
    """Exact solution of dA/dt = E, dE/dt = -A - gamma*E (linear,
    constant-coefficient 2x2 system), evaluated at times `t`.

    Characteristic equation: lambda^2 + gamma*lambda + 1 = 0
      lambda_pm = [-gamma +- sqrt(gamma^2 - 4)] / 2
    """
    disc = gamma ** 2 - 4.0
    if disc > 0:
        sq = np.sqrt(disc)
        lam1 = (-gamma + sq) / 2.0
        lam2 = (-gamma - sq) / 2.0
        # A(t) = c1*exp(lam1 t) + c2*exp(lam2 t)
        # A(0)=A0, A'(0)=E0=c1*lam1+c2*lam2, c1+c2=A0
        c1 = (E0 - lam2 * A0) / (lam1 - lam2)
        c2 = A0 - c1
        return c1 * np.exp(lam1 * t) + c2 * np.exp(lam2 * t)
    else:
        # complex / repeated-root cases not needed for gamma >> 2
        raise ValueError("gamma too small for this overdamped regime")


def adiabatic_prediction(A0, gamma, t):
    """A(t) ~= A0 * exp(-t/gamma), from dA/dt ~= -A/gamma."""
    return A0 * np.exp(-t / gamma)


def naive_wrong_prediction(A0, t):
    """The dimensionally-inconsistent guess dA/dt ~= -A (i.e. dropping
    the 1/gamma factor entirely). Included only to show it does NOT
    match, however large gamma is."""
    return A0 * np.exp(-t)


def relative_error(A0, E0, gamma, t_eval):
    A_exact = exact_A_of_t(A0, E0, gamma, t_eval)
    A_adia = adiabatic_prediction(A0, gamma, t_eval)
    A_naive = naive_wrong_prediction(A0, t_eval)
    err_adia = np.max(np.abs((A_exact - A_adia) / A0))
    err_naive = np.max(np.abs((A_exact - A_naive) / A0))
    return err_adia, err_naive


if __name__ == "__main__":
    A0, E0 = 1.0, 0.0
    # Evaluate on the SLOW time scale tau = t/gamma (i.e. physical time
    # t = tau_fixed * gamma), where the adiabatic prediction is O(1)
    # over the comparison window regardless of gamma.
    tau_fixed = np.linspace(0.05, 1.0, 20)

    print(f"{'gamma':>10} {'max rel err (adiabatic)':>26} "
          f"{'max rel err (naive dA/dt=-A)':>30}")
    for gamma in [10, 30, 100, 300, 1_000, 3_000, 10_000, 100_000]:
        t_eval = tau_fixed * gamma
        err_adia, err_naive = relative_error(A0, E0, gamma, t_eval)
        print(f"{gamma:10d} {err_adia:26.3e} {err_naive:30.3e}")

    print()
    print("Expected: err_adiabatic -> 0 as gamma increases (eventually")
    print("below 1e-8, i.e. 8 significant figures), while err_naive")
    print("stays of order 1 regardless of gamma (confirming the naive")
    print("guess dA/dt ~= -A is not the correct large-gamma limit).")
