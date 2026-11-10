"""
Reproduces the static-solution residual check of Section 4.1.

DUFT defines a static solution not by dA/dt = dE/dt = 0 alone (an
initial condition, not necessarily persistent), but by the vanishing
of the GENERALIZED FORCE that remains after substituting dA/dt=E=0,
dE/dt=0 into Hamilton's equations:

    lhs(A) := delta H / delta A

For the linear toy model used in the paper, delta H / delta A = A
(i.e. the toy "potential" is H = (1/2) A^2, so its gradient is simply
A). The check is that this residual function:

  (1) returns exactly A for arbitrary A (i.e. it genuinely computes
      the gradient rather than being hard-wired to return 0), and
  (2) is zero if and only if A = 0,

so that the function correctly *reports* the static condition instead
of presupposing it.
"""
import numpy as np


def lhs(A):
    """delta H / delta A for the linear toy model H = (1/2) A^2."""
    return A


def is_static(A, tol=1e-12):
    return abs(lhs(A)) < tol


if __name__ == "__main__":
    print("Check 1: lhs(A) returns exactly A for arbitrary A")
    rng = np.random.default_rng(0)
    test_values = np.concatenate([
        [0.0, 1.0, -1.0, 3.14159],
        rng.normal(scale=5.0, size=10),
    ])
    all_exact = True
    for A in test_values:
        residual = lhs(A)
        exact = (residual == A)
        all_exact &= exact
        print(f"  A={A:12.6f}   lhs(A)={residual:12.6f}   "
              f"exactly equal to A: {exact}")
    print(f"  -> all exact: {all_exact}")

    print()
    print("Check 2: is_static(A) is True iff A == 0")
    results = []
    for A in [0.0, 1e-13, 1e-10, 0.001, 1.0, -2.5]:
        static = is_static(A)
        results.append((A, static))
        print(f"  A={A:12.6g}   is_static={static}")
    only_zero_is_static = (
        results[0][1] is True
        and results[1][1] is True   # within tol of 0
        and all(not s for a, s in results[2:])
    )
    print(f"  -> only (numerically) zero values are reported static: "
          f"{only_zero_is_static}")
