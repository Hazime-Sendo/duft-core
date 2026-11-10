# Verification log

This document records the specific sign/normalization errors that were
identified and corrected during development, and the numerical method
used to catch each one. It is included for transparency: several of
these errors were individually plausible on inspection and were caught
only by direct numerical comparison, not by code review alone.

## 1. Magnetic energy normalization

**Incorrect candidate:** `H_B += 1.0 - (1.0/3.0)*Re(Tr(U_plaq))`

**Correct formula:** `H_B += Tr(I - U_plaq) = 3 - Re(Tr(U_plaq))`

**How it was caught:** the incorrect formula is exactly 1/3 of the
correct one. Since `force_x`/`force_y` are the variation of the
*correct* (unscaled) potential, using the 1/3-scaled energy as the
quantity expected to be conserved produces a large, dt-independent
drift (~0.2 over 200 steps at dt=0.01), while the correctly-scaled
energy is conserved to 5 significant figures under the same dynamics.

## 2. Electric energy contraction (einsum index order)

**Incorrect candidate:** `np.einsum('ijab,ijba->', Ex, Ex.conj())`

**Correct formula:** `np.einsum('ijab,ijab->', Ex, Ex.conj())`

**How it was caught:** the incorrect (swapped) index order computes
`sum E_ab * conj(E_ba)`, which for Hermitian E equals `sum E_ab^2`
(a complex square, not `sum |E_ab|^2`). Direct comparison against a
loop-based reference implementation showed a factor-of-~3
discrepancy for a specific random test configuration.

## 3. Staple/force sign convention (energy conservation)

Several structurally plausible sign conventions for the force formula
(`Omega = U @ S` vs `U @ S.conj().T`; `E -= dt*F` vs `E += dt*F`) were
tested. Only one combination -- `Omega = U @ S` (no extra dagger on
the staple) together with `E -= dt*F` -- conserves energy; the other
combinations produce either immediate divergence or convergence (under
dt refinement) to a value other than the initial energy. This was
determined by:

1. Direct finite-difference comparison: perturbing a single link
   `U(eps) = exp(i*eps*X) U0` for a random traceless-Hermitian `X`,
   and comparing `dS/deps` (numerical) against `Tr(X @ F)` for each
   candidate force formula `F`.
2. dt-independence check: running the leapfrog integrator at
   `dt = 0.02, 0.01, 0.005` and confirming that `E(t=1.0)` converges,
   as `dt -> 0`, to the *initial* energy (not to some other,
   dt-independent value, which would indicate the wrong continuum
   equations are being solved exactly).

## 4. Staple asymmetry between x- and y-directions

For the x-direction link, only the "lower" (backward-neighbor) staple
is dagger-conjugated before summing; for the y-direction link, *both*
staples are conjugated. This asymmetry was confirmed correct (not a
bug) by the finite-difference check in item 3, applied separately to
an x-link and a y-link.

## 5. Covariant Gauss's-law formula

**Incorrect candidate:**
`D_i E_i(x) = U_i(x) E_i(x+i) U_i(x)^dag - U_i(x-i)^dag E_i(x-i) U_i(x-i)`

**Correct (standard Kogut-Susskind) formula:**
`G(x) = sum_i [ E_i(x) - U_i(x-i)^dag E_i(x-i) U_i(x-i) ]`

**How it was caught:** running both formulas under the verified
leapfrog dynamics starting from a nonzero random E (so `G(0) != 0`),
the correct formula's norm `||G(t)||` was found to be *exactly*
conserved (to floating-point round-off, ~1e-15, independent of dt and
lattice size), while the incorrect formula's norm fluctuated
substantially (order-unity changes within a few hundred steps).

## 6. Perturbation direction for Lyapunov measurements

An initial choice of perturbation, `U(0,0) = exp(i*eps*I)` (a pure
U(1) phase, i.e. a scalar multiple of the identity), was found to
produce *zero* divergence between the two trajectories for the entire
run. This is not a bug: `force_x`/`force_y` project onto the
traceless subspace, so a pure-trace (U(1)) perturbation is exactly
invisible to the SU(3) dynamics. The perturbation was changed to a
genuine su(3) (traceless Hermitian, e.g. the Gell-Mann lambda_3)
direction, which produces the expected nonzero, growing divergence.

A related diagnostic error: an early version of the high-k mode
extraction took `Ax_s = trace(Ax)` before Fourier-transforming, which
is *also* exactly zero for any traceless perturbation, for the same
reason. The fix was to Fourier-transform each matrix component of
`A = (U-I)/i` separately, without taking the trace.

## 7. Symplectic Euler instability claim: not reproducible (retracted from the manuscript)

**Original, preliminary observation:** in an early, exploratory stage of
this project, a non-time-symmetric ("naive symplectic Euler") stepping
scheme appeared to produce energy growth by a factor of order 10^15
within O(1) time units, independent of how small dt was taken.

**Status: not reproduced; retracted from the main text.**
`scripts/investigate_symplectic_euler_claim.py` re-examined this using
the normalized force formulas and unitary link updates of the present
work (`src/gauge_su3.py`): both natural single-kick orderings of
symplectic Euler (momentum-first and field-first) and a fully explicit
forward-Euler scheme, scanned over both dt (at fixed integration time)
and integration time (at fixed dt) on a chaotic (amp=0.3) background.
In every case the energy error shrank monotonically as dt -> 0: the two
symplectic-Euler orderings showed small, bounded error, consistent with
standard backward-error-analysis results for first-order symplectic
integrators, and the explicit-Euler scheme showed ordinary first-order
error that grew slowly with the integration window but still vanished
as dt -> 0. No scheme tested reproduced growth independent of dt.

This strongly suggests that the original observation was an artifact of
a non-unitary update rule or an incorrect force formula used at that
early, pre-verification stage of the project, rather than a genuine
structural instability of the dynamics. The claim has accordingly been
removed from the main text of the manuscript and replaced with a note
pointing to this log entry; the manuscript also records it in
Appendix~B for transparency. None of
the manuscript's central results (convergence orders, exact Gauss's-law
conservation, adiabatic elimination, the static-residual check, or the
FRW-derived 2H friction term) depend on this retracted claim -- each is
independently verified via the time-symmetric leapfrog and fourth-order
Yoshida integrators instead.

## 8. Link-based low-wavenumber growth: gauge-dependent, interpretation withdrawn

**Original reading (earlier draft):** the exponent measured from low-wavenumber
modes of the Fourier-transformed link variables, lambda_low, grew first and
saturated near 0.28 while the high-wavenumber exponent stayed small, which was
read as a two-stage transition ("chaos first in the long-wavelength sector").

**Check:** the same pair of trajectories was re-measured at paper scale
(32x32, T=10; `scripts/lyapunov_paper_scale.py`) with a second, gauge-invariant
distance built from the Wilson-plaquette field. At amp=0.05 the link-based
low-k distance grows (lambda ~ 0.16) while the plaquette distance does not
(lambda ~ 0.001); the high-k link distance and the plaquette distance agree
at every amplitude. lambda_low also depends strongly on the fit window
(amp=0.3: 0.16 to 0.34 for fit start t0=6 and 1) and on lattice size.

**Follow-up with four seeds (amp 0.1 and 0.2; seeds 42, 1, 2, 3):** the
gauge-invariant low-k plaquette component grows faster than the high-k one at
amp=0.1 in all three windows and all four seeds (diff +0.05 to +0.11), and at
amp=0.2 in windows 3-7 and 5-9 (4/4), with the order reversed in window 7-10
(0/4): a long-wavelength-first tendency with the high-k part catching up.
Caveats: the low-k component is only ~10-30 % of the high-k amplitude at t=10;
some low-k fits at amp=0.1 are poor (R^2 down to 0.3); the perturbation is a
single localized link, so spreading from a localized seed is not separated from
a dynamical mechanism; amp=0.05 was run for seed 42 only (no sustained growth).
(`scripts/analyze_lyapunov_windows.py --summary`, results/lyapunov_multiseed_summary.log)

**Status:** the link-based two-stage interpretation is not adopted. The
long-wavelength-first picture is weakly supported by gauge-invariant data at
amp 0.1-0.2, not established. (Earlier versions of this entry and of the
manuscript first said the gauge-invariant measure showed no low-k stage, then
that the question was open; the four-seed data above supersede both.)

## 9. Other corrections made after independent review

- Proposition 1 (dissipative limit): H = 1/2 Tr E^2 + 1/2 Tr B^2 gives
  dA/dtau = -1/2 delta Tr(B^2)/delta A; the identification with S_Omega holds
  up to the constant time rescaling tau' = tau/2. Statement corrected.
- Convergence table (Sec. 3.3) replaced by the values the shipped code
  outputs (e.g. 9.22e-2, not 9.03e-2 at dt=0.05); the ratios are unchanged.
- Energy drift (Sec. 5.4): the earlier "0.00389-0.00625 %" were values at
  t=T; the maximum along the trajectories is 0.0094-0.0117 %.
- Gauss's law (Sec. 6.3): the deviation does not grow in proportion to the
  number of lattice sites; the claim was removed, as was the statement about
  coupling to a gravitational sector (not implemented in this repository).
- Static-residual script (Sec. 4.1) and the FRW script (App. A) are code /
  algebra consistency checks; the manuscript wording now says so.

## 10. Additional check (candidate, not in the manuscript): random all-link perturbation

**Purpose:** item 8 left open whether the low-k lead of the gauge-invariant
plaquette distance is dynamical or an artefact of spreading from a perturbation
localized on one link. Test: perturb EVERY link by a random su(3) rotation
(`scripts/lyapunov_paper_scale.py --random`), with the total perturbation norm
equal to the single-link one (initial plaquette distance 2.0e-8 in both), same
background per seed. 32x32, T=10, amp 0.1 and 0.2, seeds 42, 1, 2, 3
(`results/lyapunov_amp*_random*.csv`, `results/lyapunov_random_vs_local_summary.log`).
Criterion fixed beforehand: low-k lead survives -> dynamical origin more likely;
lead disappears -> localized-seed artefact more likely.

**Result (plaquette distance, lambda_low - lambda_high, mean +- sd over 4 seeds):**

| amp | window | random (all links) | single link | positive seeds (random) |
|---|---|---|---|---|
| 0.1 | 3-7 | +0.084 +- 0.015 | +0.105 +- 0.042 | 4/4 |
| 0.1 | 5-9 | +0.029 +- 0.007 | +0.049 +- 0.026 | 4/4 |
| 0.1 | 7-10 | +0.078 +- 0.011 | +0.096 +- 0.035 | 4/4 |
| 0.2 | 3-7 | +0.124 +- 0.018 | +0.154 +- 0.031 | 4/4 |
| 0.2 | 5-9 | +0.064 +- 0.021 | +0.047 +- 0.070 | 4/4 |
| 0.2 | 7-10 | +0.030 +- 0.016 | -0.066 +- 0.054 | 4/4 (single link: 0/4) |

**Reading:** with the random perturbation the low-k lead persists in every
window, for every seed, at both amplitudes; in the early and middle windows its
size is within seed scatter of the single-link case (paired differences
-0.02 to -0.03 at amp 0.1 and 0.2 early; see the summary log). This
strengthens the possibility that the lead is dynamical rather than a
localization artefact. The one clear difference is amp=0.2, window 7-10: the
"high-k catches up" reversal seen with the single-link perturbation (-0.066)
is absent with the random one (+0.030), so that part is more likely an effect
of the localized seed.

**Caveats:** (i) both perturbations have a flat link-variable spectrum and equal
norm; they differ in spatial coherence, so this removes localization but is not
a test of seeding a specific shell (a perturbation restricted to high-k only
would be a sharper test); (ii) the lead is modest (lambda difference 0.03-0.12)
and the low-k component stays small (lo/hi amplitude ratio 0.08-0.14 at t=0,
0.14-0.29 at t=10); (iii) low-k fits at amp 0.1 are poor in places (R^2 down to
0.56); (iv) 4 seeds, one lattice size, T=10, amp 0.05 and 0.3 not run with the
random perturbation. Manuscript wording ("weak support") is unchanged.
(Superseded in part by item 11: shell-targeted perturbations show that the lead
is seed-spectrum dependent.)

## 11. Additional check (candidate, not in the manuscript): shell-targeted perturbations, amplitude scan, finer windows

**Purpose:** try to settle caveat (iii) of the manuscript (low-k lead of the
gauge-invariant plaquette distance may be an artefact of the perturbation's
spatial structure) with three tests fixed beforehand: (1) perturb only one |k|
shell (criterion: the low-k lead appears whichever shell is seeded); (2) random
perturbation at amp 0.05 and 0.3 as well (criterion: sign positive at every
amplitude); (3) finer windows 2-5, 4-7, 6-9, 8-10 (criterion: sign positive in
every window). All runs 32x32, dt 0.01, T=10, same backgrounds per seed.

**Method.** `scripts/lyapunov_paper_scale.py --shell low|mid|high`: a white random
traceless-Hermitian field on all links (Ux and Uy), filtered in Fourier space to
|k|<4 (low), 4<=|k|<8 (mid) or |k|>=8 (high) (the diagnostic split), rescaled to
the same total link-perturbation norm as the single-link/random cases
(1e-8*sqrt 2), links -> exp(iX)U; RNG stream [seed,2]. Runs: shells x amp 0.1, 0.2 x
seeds 1,2,3,42 (24 runs); random perturbation amp 0.05, 0.3 x seeds 1,2,3 (6 runs).
Because the link norm is matched, the INITIAL plaquette distance differs by shell
(seed 1, amp 0.1: low 0.76e-8, mid 1.24e-8, high 2.14e-8; random/single-link 2.0e-8);
in the linear regime the exponents do not depend on the initial scale. Outputs:
`results/lyapunov_amp*_shell*.csv|log`, `lyapunov_amp{0.05,0.3}_random_seed*.csv|log`,
`lyapunov_shell_summary.log`, `lyapunov_fine_windows_summary.log`,
`lyapunov_windows_shell_ampscan.log`, `lyapunov_windows_fine.log`,
`lyapunov_spectral_ratio.log` (`scripts/analyze_spectral_ratio.py`).

**Result 1 - shells (lambda_low - lambda_high, plaquette distance; mean over seeds 1,2,3,42; positive seeds in brackets).**

| amp | window | shell-low | shell-mid | shell-high | random | single link |
|---|---|---|---|---|---|---|
| 0.1 | 3-7 | -0.033 (0/4) | +0.046 (4/4) | +0.093 (4/4) | +0.084 (4/4) | +0.105 (4/4) |
| 0.1 | 5-9 | -0.143 (0/4) | +0.007 (3/4) | +0.060 (4/4) | +0.029 (4/4) | +0.049 (4/4) |
| 0.1 | 7-10 | +0.029 (4/4) | +0.016 (3/4) | +0.070 (4/4) | +0.078 (4/4) | +0.096 (4/4) |
| 0.2 | 3-7 | -0.011 (1/4) | +0.056 (4/4) | +0.133 (4/4) | +0.124 (4/4) | +0.154 (4/4) |
| 0.2 | 5-9 | -0.045 (0/4) | +0.001 (2/4) | +0.086 (4/4) | +0.064 (4/4) | +0.047 (4/4) |
| 0.2 | 7-10 | +0.003 (3/4) | -0.014 (0/4) | +0.038 (4/4) | +0.030 (4/4) | -0.066 (0/4) |

The criterion is NOT met: seeding the low-k shell removes the lead (negative in
windows 3-7 and 5-9 at both amplitudes, 0/4 seeds positive in 5-9), seeding the mid
shell gives a weak/inconsistent lead, seeding the high shell gives a lead
comparable to the random and single-link cases (paired high-shell minus random:
-0.01 .. +0.03, within scatter).

**Result 2 - amplitude scan (random perturbation, seeds 1,2,3 for every amplitude).**

| amp | 3-7 | 5-9 | 7-10 |
|---|---|---|---|
| 0.05 | +0.071 +- 0.012 (3/3) | -0.026 +- 0.007 (0/3) | +0.092 +- 0.014 (3/3) |
| 0.1 | +0.084 +- 0.018 (3/3) | +0.032 +- 0.003 (3/3) | +0.075 +- 0.011 (3/3) |
| 0.2 | +0.127 +- 0.021 (3/3) | +0.057 +- 0.019 (3/3) | +0.024 +- 0.015 (3/3) |
| 0.3 | +0.060 +- 0.005 (3/3) | -0.015 +- 0.011 (0/3) | -0.001 +- 0.021 (1/3) |

Window 3-7 is positive at all four amplitudes, but the sign is NOT positive at
every amplitude in windows 5-9 (amp 0.05, 0.3) and 7-10 (amp 0.3).
Criterion not met in full; the magnitude varies smoothly with amplitude in 3-7.

**Result 3 - finer windows (mean over seeds; positive seeds in brackets).**

| amp | perturbation | 2-5 | 4-7 | 6-9 | 8-10 |
|---|---|---|---|---|---|
| 0.05 | local | +0.294 (1/1) | -0.066 (0/1) | +0.053 (1/1) | +0.122 (1/1) |
| 0.05 | random | +0.254 (3/3) | -0.042 (0/3) | +0.022 (3/3) | +0.108 (3/3) |
| 0.1 | local | +0.220 (4/4) | +0.027 (3/4) | +0.079 (4/4) | +0.091 (4/4) |
| 0.1 | random | +0.188 (4/4) | +0.019 (4/4) | +0.053 (4/4) | +0.087 (4/4) |
| 0.1 | shell-low | +0.245 (4/4) | -0.179 (0/4) | -0.072 (0/4) | +0.015 (2/4) |
| 0.1 | shell-mid | +0.069 (4/4) | +0.002 (1/4) | +0.017 (3/4) | +0.020 (3/4) |
| 0.1 | shell-high | +0.148 (4/4) | +0.065 (4/4) | +0.072 (4/4) | +0.078 (4/4) |
| 0.2 | local | +0.169 (4/4) | +0.151 (4/4) | -0.000 (2/4) | -0.078 (0/4) |
| 0.2 | random | +0.173 (4/4) | +0.093 (4/4) | +0.055 (4/4) | +0.017 (4/4) |
| 0.2 | shell-low | +0.151 (4/4) | -0.094 (0/4) | -0.008 (1/4) | +0.012 (3/4) |
| 0.2 | shell-mid | +0.086 (4/4) | +0.015 (3/4) | -0.007 (1/4) | +0.018 (4/4) |
| 0.2 | shell-high | +0.172 (4/4) | +0.110 (4/4) | +0.081 (4/4) | +0.033 (3/4) |
| 0.3 | local | +0.145 (1/1) | -0.051 (0/1) | -0.022 (0/1) | +0.060 (1/1) |
| 0.3 | random | +0.150 (3/3) | +0.022 (3/3) | -0.025 (0/3) | +0.010 (3/3) |

Window 2-5 is positive in every case (all perturbation types incl. shell-low,
all amplitudes). Windows 4-7 and 6-9 change sign depending on perturbation type
and amplitude (shell-low: negative; random amp 0.05: negative in 4-7; amp 0.3:
negative in 6-9). Criterion not met in full.

**Mechanism check (existing data, no new runs).** `lyapunov_spectral_ratio.log`:
the ratio R(t) = plaq_lo/plaq_hi (geometric mean over seeds) moves toward a
common value from either side. amp 0.2: t=0 -> t=10: shell-low 0.43 -> 0.30,
shell-mid 0.18 -> 0.31, shell-high 0.12 -> 0.28, random 0.13 -> 0.28, single link
0.14 -> 0.25. amp 0.1 (not converged by t=10): shell-low 0.88 -> 0.41, shell-mid
0.17 -> 0.28, shell-high 0.06 -> 0.13, random 0.08 -> 0.15. The total plaquette
distance changes little (t=10 / t=0: 0.7-0.8 at amp 0.05-0.1 for random,
single-link and shell-high; 1.5-3.0 at amp 0.2 and for shell-low/mid; 4.7-6.4 at amp 0.3),
i.e. much of what the window-wise exponent difference measures is redistribution
of the perturbation between low and high k, not exponential instability.

**Reading.** (a) Items 10 and 11 together show that the lead does not require a
localized seed (random and high-shell seeds reproduce it) - this part of caveat
(iii) is reduced. (b) But the lead is not a seed-independent generation
direction: it flips sign when the low-k shell is seeded, and the sign pattern is
consistent with relaxation of the low/high ratio toward a common, amplitude-
dependent value: a component that starts relatively small grows faster, one that
starts relatively large grows slower. Generic (white-in-link) perturbations all
start below that value, hence a "low-k lead". (c) The statement made after item
10 ("the possibility of a dynamical long-wavelength-first lead is strengthened")
should therefore be read as: the lead is dynamical in the sense of spectral
relaxation, and is NOT evidence for an intrinsic long-wavelength-first direction.
(d) The pre-set criteria of tests 1-3 were not met; caveat (iii) is not resolved
to the 70-80% level.

**Caveats:** one lattice size (32x32), T=10, 4 seeds (3 for amp 0.05, 0.3), unmodified
leapfrog; ratio convergence is incomplete at amp 0.1; plaq_lo/plaq_hi are
unnormalised Fourier-amplitude norms of the plaquette difference over unequal
numbers of modes, so only the time-dependence of R, not its absolute value,
is meaningful; fits over short windows (3-5 time units) have large scatter and
some low R^2; the mid shell has no matching diagnostic band (4<=|k|<8 is
excluded from lo/hi). Manuscript wording ("weak support") unchanged; if the text
is revised, a statement that the lead depends on the initial perturbation spectrum
would be warranted.

## 12. Additional check (candidate, not in the manuscript): the relaxation model, tested directly

**Hypothesis (formed after item 11):** the low/high-k split of the plaquette
perturbation is not a "long-wavelength-first generation direction" but a
relaxation toward a common, amplitude-dependent spectrum: components that start
small grow faster, large ones slower; lambda_low - lambda_high = d ln R / dt.
**Tests fixed beforehand:** (1) long times T=20-40 (does R converge? does the sign
of lambda_low - lambda_high change?); (2) lattice size 64x64, 128x128 (same
common value?); (3) artificially prepared initial spectra (low-heavy, high-heavy,
flat, mid-shell) - same final ratio?

**Method.** `scripts/lyapunov_paper_scale.py` extended: `--n` (lattice), `--T`,
`--spectrum lowpeak|midpeak|highpeak|flat` (link-perturbation weights: |k|<2;
5<=|k|<7; |k|>=12; 1/lattice-curl factor, i.e. approximately white in the
plaquette), `--engine fast|ref`; it now also writes `plaq_mid` and the
shell-resolved spectrum (`results/spectrum_*.csv`, mean power per mode per integer
|k| bin, every 1 time unit) and its own `.log`. Total link-perturbation norm is
the same in all runs (1e-8*sqrt 2), so the initial plaquette distance differs
slightly by spectrum (exponents in the linear regime do not depend on it).
Quantity: Rpm(t) = sqrt[(plaq_lo^2/n_lo)/(plaq_hi^2/n_hi)], the per-mode amplitude
ratio of the low-k (|k|<4) to the high-k (|k|>=8) plaquette-difference
(the raw ratio plaq_lo/plaq_hi contains a trivial lattice-size factor ~N; earlier
items used the raw ratio). Rpm = 1 means equal power per Fourier mode.
Runs (60): N=32, T=40, seeds 1-3: random perturbation at amp 0.05, 0.1, 0.2, 0.3;
shell-low and shell-high, and the four artificial spectra, at amp 0.1, 0.2;
N=64, T=40 and N=128, T=20, random perturbation, amp 0.1, 0.2, seeds 1-3.
Output files `results/lyapunov_N<n>_T<T>_amp*.csv|log`, `spectrum_*.csv`,
`lyapunov_relaxation_summary.log`, `relax_fig1_ratio_vs_time.png`,
`relax_fig2_lattice_size.png`, `relax_fig3_spectra.png`
(`scripts/analyze_relaxation.py`).

**Speed-up and validation of the engine.** These runs use the vectorised leapfrog
(`src/gauge_su3_fast.py`, same formulas, ~10x faster; the reference loops in
`src/gauge_su3.py` remain authoritative and were used for items 1-11).
`scripts/verify_fast_engine.py`: after 100 steps max |dU| = 1.5e-14, energy equal to
1e-12 relative. Re-running the item-10 case (amp 0.1, random, seed 1, 32x32, T=10)
with the fast engine reproduces the stored distances to <=1.7e-5 relative and all
exponents to 4 digits (`results/engine_check/`).

**Result 1 - long time (N=32, amp 0.1; Rpm geometric mean of seeds 1-3).**

| initial spectrum | t=0 | t=10 | t=20 | t=40 |
|---|---|---|---|---|
| shell-low | 3.86 | 1.81 | 1.06 | 1.09 |
| lowpeak | 1.70 | 0.92 | 0.96 | 1.10 |
| midpeak | 0.71 | 1.19 | 1.04 | 1.09 |
| flat (plaquette) | 0.95 | 0.99 | 0.96 | 1.10 |
| random (white) | 0.36 | 0.64 | 0.89 | 1.06 |
| shell-high | 0.28 | 0.56 | 0.90 | 1.03 |
| highpeak | 0.28 | 0.49 | 0.83 | 1.10 |

Spread across the 7 initial spectra (max/min of Rpm): 14.0 at t=0 (initial
range 0.28-3.86), 3.7 at t=10, 1.28 at t=20, **1.07 at t=40**. At amp 0.2 the seven
spectra start within 0.53-1.87 and end within 1.08-1.12 (spread 1.04 at t=40).
The shell-resolved spectra (fig. 3) differ by up to two orders of magnitude at
t=0 and collapse at t=40 onto one nearly flat per-mode spectrum (max deviation in
ln S over bins 1-15: 0.12 at amp 0.1, 0.13 at amp 0.2; at t=0: 3.0 and 1.8).
lambda_low - lambda_high (= d ln R/dt) changes sign with time for every
spectrum that starts above/below the common value (shell-low: negative for
t<20, then ~0; random/shell-high/highpeak: positive, then ~0; amp 0.2 random:
+0.095 (0-5), +0.046 (5-10), then 0 to slightly negative: -0.009..-0.005) and goes to ~0 (|slope|<=0.01 in
windows 20-40) for all.

**Result 2 - amplitude dependence (random, N=32, T=40).** Rpm(40): 0.58 (amp
0.05, NOT converged), 1.06 (0.1), 1.10 (0.2), 1.01 (0.3). Fitted
ln R = A + B exp(-gamma t): gamma = 0.076 (0.1), 0.28 (0.2), 0.61 (0.3), i.e.
gamma ~ amp^1.9 (two successive log-log slopes 1.89 and 1.90); asymptote R*:
1.15, 1.16, 1.06 (weakly amplitude dependent). amp 0.05 fit is unreliable
(R* extrapolated to 10.6, gamma 0.007; expected gamma ~ 0.02 from the amp^2 trend, which
would need T >> 40).

**Result 3 - lattice size (random).** amp 0.2: R(10) 1.20/1.25/1.22,
R(20) 1.18/1.13/1.16 for N = 32/64/128; fitted R* 1.27/1.26/1.24, gamma
0.22/0.24/0.24: no lattice dependence within seed scatter. amp 0.1: R(40) = 1.056
(N=32), 1.054 (N=64) - same common value; but the approach is faster on larger
lattices: R(10) = 0.64/0.82/0.84, R(20) = 0.89/0.97/1.10 for N = 32/64/128
(seed sd 0.02-0.09); N=128 overshoots (1.18 at t=15) and was run only to T=20.
No sign instability: slope(5-10) positive for all 3 seeds at every N, slope(10-20) ~0.

**Reading.** Within this ensemble the relaxation model is strongly supported:
(a) very different initial spectra converge to the same Rpm (~1.05-1.1) and the same
flat per-mode spectrum, (b) the sign of lambda_low - lambda_high is set by whether
the initial ratio is below or above that value and vanishes at late times,
(c) the common value does not depend on lattice size (32, 64 [128 to T=20]) and
only weakly on amplitude; the relaxation RATE scales ~amp^1.9. Items 8-11 are
then explained: a low-k lead appears for every perturbation that starts with
Rpm < ~1.1 (white-in-link, localized, high-shell) and disappears or reverses for
those that start above it. The common value Rpm ~ 1 means equal power per mode
(equipartition of the perturbation between low and high k). Items 8-11 do not
support an intrinsic long-wavelength-first generation direction.

**Caveats (important).** (i) The backgrounds are white-noise random link
configurations (strongly disordered; not a DUFT classical solution or a smooth
vacuum). A flat per-mode final spectrum and a rate ~ amp^2 are what one expects
for linear scattering of a perturbation in a disordered medium, so the "common
spectrum" may be a property of this white-noise background ensemble rather than
of DUFT dynamics in general; smooth/structured backgrounds were NOT tested and
could give a different attractor. (ii) The total plaquette distance grows only
mildly, so this is mostly redistribution of the perturbation among k, not
exponential instability. (iii) Seeds 1-3 (3 per condition); amp 0.05 not converged;
N=128 only to T=20; k-bands are fixed integer-|k| cuts, not fixed physical
wavelengths; early transient spike of R at t~0.7 in all runs. (iv) The
exponential form A + B exp(-gamma t) is a convenient description (rms residual
0.05-0.16 in ln R), not a derived law; gamma is ill-defined when the initial value
is already near R*. (v) Manuscript unchanged; if the text is revised, the
statement should be that the low-k lead reflects relaxation of the initial
perturbation spectrum toward an equal-power-per-mode spectrum in white random backgrounds.

## 13. Additional check (candidate, not in the manuscript): does the relaxation depend on the background? (smooth and low-k-strong backgrounds)

**Purpose:** item 12 supported relaxation to a flat per-mode spectrum, but only
for white-noise random backgrounds. Same three tests (long time, lattice size,
artificial initial spectra) in two further backgrounds, with Rpm convergence and spectral flatness
as the metrics. `scripts/lyapunov_paper_scale.py --bg smooth|lowstrong`
(`make_background`): **smooth** = the same random su(3) field (same RNG draws)
low-pass filtered to |k|<=2, rescaled to the same rms per link, U=exp(i amp H), E=0
(nearly flat, slowly varying; NOT an exact classical solution); **lowstrong** = the
white field with its |k|<4 modes multiplied by 3 (power x9). Background strength
mean(1 - ReTr P/3) (seed 1, N=32): white 0.051 / 0.19 (amp 0.1 / 0.2); smooth
0.0016 / 0.014 / 0.28 (amp 0.1 / 0.2 / 0.5); lowstrong 0.052 / 0.20. Because smoothness
and field strength are confounded at equal amp, smooth amp 0.5 (field strength
comparable to white amp 0.2-0.3) was added. Runs (89): N=32, T=40, seeds 1-3:
random + lowpeak/midpeak/highpeak/flat perturbations, amp 0.1, 0.2, both backgrounds
(shell-low/high not repeated); smooth amp 0.5: random, lowpeak, highpeak; N=64 T=40
(seeds 1-3) and N=128 T=20 (seeds 1-2 only, to save time), random, amp 0.1, 0.2, both
backgrounds. Files: `results/lyapunov_N*_T*_amp*_bg{smooth,lowstrong}_*.csv|log`,
`spectrum_*_bg*.csv`, `lyapunov_relaxation_backgrounds_summary.log`,
`relax_{smooth,lowstrong}_fig{1,2,3}_*.png`, `relax_fig4_backgrounds.png`
(`scripts/analyze_relaxation.py`, `scripts/plot_relaxation_backgrounds.py`).

**Result A - low-k-strong background: same as white.** All 5 initial spectra converge:
spread (max/min of Rpm) at T=40 = 1.07 (amp 0.1; 1.04 at amp 0.2) against 2.3 / 1.10 at t=10; Rpm(40) =
1.13-1.21 (amp 0.1), 1.08-1.13 (amp 0.2); final spectra flat (std of ln S over |k| bins 1-15: 0.12-0.16;
white 0.10-0.13; power ratio bins 1-3 / bins 12-15: 1.2-1.45; white 1.0-1.1); shell-resolved collapse 0.10-0.15 in ln S.
Rate: gamma = 0.11 (amp 0.1; white 0.076), 0.31 (amp 0.2; white 0.28). Lattice size: amp 0.2 identical
(R(20) 1.16/1.21/1.18 for N=32/64/128), amp 0.1 as in white (faster approach on larger N).
So a strong long-wavelength component does not change the attractor (equal power per mode).

**Result B - smooth, weak-field background: no common value within T=40.**
- amp 0.1 (field strength 0.0016): Rpm(40) for the five spectra = 12.2 (lowpeak),
5.3 (midpeak), 7.7 (flat), 2.0 (random), 0.026 (highpeak); spread 473 (initial 1e8),
none flat (std ln S 0.7-3.3; low/high power ratio 5 ... 2000 ... 0.00); the high-k-seeded
run is still rising (lambda_low - lambda_high = +0.043 in 30-40) and the low-k-seeded one still
falling. No convergence; exponential fits meaningless (flagged in the summary).
- amp 0.2 (0.014): lowpeak, midpeak, flat, random all arrive at Rpm = 2.72-2.82 by t=40
(common value, but NOT 1 and NOT flat: low/high power ratio ~10.4-11.3, std ln S 0.97-1.0);
highpeak is still rising (1.42, slope +0.083 in 30-40; overall spread 2.0).
- amp 0.5 (0.28, strength comparable to white): all three tested spectra converge to
Rpm = 1.01-1.07 with flat spectra (std ln S 0.09-0.10, power ratio 0.98-1.20), rate gamma 0.3-0.7.
So relaxation to the flat per-mode spectrum needs a sufficiently strong (mixing) background field;
smoothness of the background as such is not what prevents it (amp 0.5 smooth equilibrates).
In the weak smooth background the perturbation spectrum is only slowly redistributed
(near-free propagation) and the late-time lead sign is set by the initial spectrum.
- Lattice size (smooth, random): sign of lambda_low - lambda_high stable (slope 10-20 positive for
all seeds at N=32/64/128; amp 0.2 R(20) 2.22/1.97/2.07); at amp 0.1 R(20) = 0.60/0.55/0.41 differs
because R0 itself depends on N (0.28/0.22/0.19).

**Reading.** The relaxation toward a common, flat (equal power per mode) spectrum is robust to a strong long-wavelength
background component, but it is a property of backgrounds with sufficiently strong field (strong mode mixing):
in nearly flat smooth backgrounds it is slow or absent within T=40, and the quasi-common value found at amp 0.2
(Rpm~2.8) is not equipartition. Therefore the "common spectrum" is not a universal attractor of DUFT dynamics: it is
conditional on field strength/disorder. The earlier conclusion that no intrinsic long-wavelength-first generation
direction is visible is unchanged; what is added: in weak smooth backgrounds neither a common value nor a generic low-k lead exists
(the sign follows the initial spectrum: low-k-seeded runs have lambda_low - lambda_high < 0, high-k-seeded > 0 for t up to 40).

**Caveats.** (i) The smooth background is a low-pass-filtered random field, not an exact classical solution;
true solutions (e.g. exactly flat or instanton-like configurations) were not tested. (ii) Strength and smoothness
are separated only by the single extra amplitude 0.5 (3 spectra); no scan of the cut |k|<=2. (iii) 3 seeds (2 at N=128, T=20),
shell-low/high and amp 0.05/0.3 not repeated in the new backgrounds; amp 0.1 smooth is far from converged, so
"no convergence" means within T=40 (a much longer run could still converge). (iv) lowstrong boosts only |k|<4 by x3;
other profiles untested. (v) Rpm uses fixed integer-|k| bands. Manuscript unchanged.

## Note added in v1.0.1: Gauss's-law check is now component-wise

`verify_gauss_law.py` previously reported only the norm deviation
| ||G(t)|| - ||G(0)|| |. Conservation of the norm does not imply
conservation of each component. The script now also reports the maximum
component-wise deviation max |G_ab(x,t) - G_ab(x,0)| over all sites and matrix
elements (new function `covariant_gauss_field` in `src/gauge_su3.py`;
`covariant_gauss_residual` is unchanged in definition). Result: component
deviations <= 5.8e-15 for all lattices (6x6 to 64x64) and time steps tested, i.e. round-off
level. The round-off-level values of the norm deviation depend on summation
order (the refactoring changed the order), and are only meaningful as orders
of magnitude. No analytic proof of exact conservation is claimed.
