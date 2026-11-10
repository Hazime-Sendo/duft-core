"""
Reproduces the first-principles derivation of Section 3 (and the Appendix): starting from
the (2+1)-dimensional FRW metric, compute the Christoffel symbols and
the covariant d'Alembertian, and show that the homogeneous-mode
equation matches the DUFT gravitational-sector update equation with
Hubble friction coefficient 2H (not 3H, which is the (3+1)-dimensional
value).
"""
import sympy as sp


def main():
    t, x, y = sp.symbols("t x y", real=True)
    a = sp.Function("a")(t)
    Psi = sp.Function("Psi")(t, x, y)

    g = sp.Matrix([[-1, 0, 0], [0, a ** 2, 0], [0, 0, a ** 2]])
    g_inv = g.inv()
    coords = [t, x, y]

    def christoffel(l, m, n):
        result = 0
        for r in range(3):
            term = (sp.diff(g[r, n], coords[m]) + sp.diff(g[r, m], coords[n])
                    - sp.diff(g[m, n], coords[r]))
            result += g_inv[l, r] * term
        return sp.simplify(result / 2)

    def box(field):
        result = 0
        for mu in range(3):
            for nu in range(3):
                second = sp.diff(field, coords[mu], coords[nu])
                gamma_term = sum(
                    christoffel(rho, mu, nu) * sp.diff(field, coords[rho])
                    for rho in range(3)
                )
                result += g_inv[mu, nu] * (second - gamma_term)
        return sp.simplify(result)

    Box_Psi = box(Psi)
    print("Full covariant d'Alembertian:")
    sp.pprint(Box_Psi)

    Psi_hom = sp.Function("Psi_hom")(t)
    Box_Psi_hom = Box_Psi.subs({
        sp.diff(Psi, x): 0, sp.diff(Psi, y): 0,
        sp.diff(Psi, x, x): 0, sp.diff(Psi, y, y): 0,
        Psi: Psi_hom,
    })
    print("\nHomogeneous-mode reduction:")
    sp.pprint(Box_Psi_hom)

    H = sp.diff(a, t) / a
    V = sp.Function("V")(Psi_hom)
    Vprime = sp.diff(V, Psi_hom)

    KG_eq = sp.simplify(Box_Psi_hom - Vprime)
    DUFT_update = sp.simplify(
        sp.diff(Psi_hom, t, 2) + 2 * H * sp.diff(Psi_hom, t) + Vprime)

    print("\nKG_eq + DUFT_update (should be 0, i.e. KG_eq = -DUFT_update: "
          "the two equations differ by the factor -1):")
    sp.pprint(sp.simplify(KG_eq + DUFT_update))


if __name__ == "__main__":
    main()
