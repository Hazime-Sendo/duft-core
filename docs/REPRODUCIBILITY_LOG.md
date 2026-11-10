# Reproducibility log

Full run of `bash scripts/run_all.sh` (all seven scripts, exit code 0 each), plus the separate paper-scale runs of `lyapunov_paper_scale.py`.
Raw console output: `results/*.log`; environment: `results/environment.txt`.

- Environment: Python 3.11.15, numpy 2.4.4, scipy 1.17.1, sympy 1.14.0, Linux x86_64 (single machine, CPU only)
- Total wall-clock time: about 8 min 10 s
- Random seed: 42 everywhere (`numpy.random.default_rng`), so results are deterministic on a given platform; last digits of floating-point output may differ across BLAS/CPU builds.

| Paper section | Script | Result in this run | Matches paper claim? |
|---|---|---|---|
| §3 convergence | `verify_energy_convergence.py` | Halving dt: leapfrog ratio 3.99 / 4.00 / 4.00; Yoshida-4 ratio 15.87 / 15.97 / 16.00 (asymptotic). First pair (dt 0.05 to 0.02) is not a halving and is pre-asymptotic: 5.82 / 36.20 | Yes |
| §6 Gauss's law | `verify_gauss_law.py` | norm drift ≤3.6e-15 and component-wise (every matrix element, every site) drift ≤5.8e-15 for 6x6, 12x12, 32x32, 64x64 and dt = 0.02, 0.01, 0.005 (T=2) | Yes (round-off level) |
| §5 perturbation growth | `lyapunov_paper_scale.py` (32x32, T=10, seed 42; run separately for amp 0.05/0.1/0.2/0.3) | Plaquette (gauge-invariant) exponent, fit from t0=3: 0.001 / 0.022 / 0.175 / 0.278; high-k link: 0.002 / 0.018 / 0.141 / 0.300; low-k link: 0.159 / 0.231 / 0.281 / 0.264 (strongly window dependent). Energy drift max 0.0094-0.0117 % (at t=T: 0.0039-0.0063 %) | Partly. Link-based "two-stage" reading not adopted; gauge-invariant low-k-first tendency at amp 0.1-0.2 in 4/4 seeds (weak, see VERIFICATION_LOG item 8; `analyze_lyapunov_windows.py --summary`) |
| §5 smoke test | `lyapunov_phase_diagram.py` (16x16, T=5; in run_all.sh) | amp 0.05: lambda_high 0.017; amp 0.30: 0.235 | Qualitative only |
| §3/App. A FRW | `frw_klein_gordon_derivation.py` | KG_eq + DUFT_update = 0 (sympy) | Yes (2H friction) |
| §2.3 adiabatic | `verify_adiabatic_elimination.py` | Adiabatic error falls as 1/gamma^2 down to 9.0e-9 at gamma=1e4; naive guess stays about 0.95 | Yes. Note: at gamma=1e5 the error rises to 1.2e-7, a floating-point cancellation artifact of the closed-form solution, not a failure of the approximation |
| §4.1 static residual | `verify_static_residual.py` | lhs(A) = A exactly; static iff A = 0 | Yes (toy-model check only) |
| §3.4/6.1/App. B | `investigate_symplectic_euler_claim.py` | Symplectic Euler (A, B): bounded error, shrinks with dt. Explicit Euler (C): first-order error, shrinks with dt (H_end/H0 = 1.43 at dt=0.05 down to 1.007 at dt=0.001) | **Original "10^15, dt-independent" claim NOT reproduced; retracted** (see `VERIFICATION_LOG.md` item 7) |

| §5 random perturbation (not in manuscript) | `lyapunov_paper_scale.py --random` (amp 0.1, 0.2; seeds 42,1,2,3) | Low-k lead (plaquette) persists: lambda_low - lambda_high > 0 in all windows and all 4 seeds; amp 0.2 window 7-10: +0.030 (random) vs -0.066 (single link) | See VERIFICATION_LOG item 10 |
| §5 shell-targeted perturbation, amp scan, finer windows (not in manuscript) | `lyapunov_paper_scale.py --shell low/mid/high` (amp 0.1, 0.2; seeds 42,1,2,3); `--random` at amp 0.05, 0.3 (seeds 1,2,3); `analyze_lyapunov_windows.py --summary [--fine]`; `analyze_spectral_ratio.py` | Low-k lead is NOT shell-independent: shell-low seeding gives lambda_low - lambda_high < 0 in windows 3-7/5-9; shell-high reproduces the random/single-link lead; sign not positive at all amplitudes/windows; low/high ratio relaxes to a common value | See VERIFICATION_LOG item 11 |
| §5 relaxation tests (not in manuscript) | `lyapunov_paper_scale.py --T 40`, `--n 64 --T 40`, `--n 128 --T 20`, `--spectrum lowpeak/midpeak/highpeak/flat` (fast engine, `src/gauge_su3_fast.py`); `analyze_relaxation.py` | Per-mode low/high ratio of the plaquette perturbation converges to ~1.05-1.1 (flat per-mode spectrum) from all initial spectra at amp 0.1-0.3; rate ~ amp^1.9; same value at N=32, 64; amp 0.05 not converged by T=40 | See VERIFICATION_LOG item 12 |
| §5 background dependence of the relaxation (not in manuscript) | `lyapunov_paper_scale.py --bg smooth/lowstrong` (+ --T 40, --n 64/128, --spectrum), `analyze_relaxation.py`, `plot_relaxation_backgrounds.py` | Low-k-strong background: same flat equipartition (Rpm~1.1-1.2) as white; smooth weak-field background (amp 0.1, 0.2): no common value/flatness within T=40 (amp 0.2: Rpm~2.8 plateau for four spectra, not flat); smooth amp 0.5 (strong field): flat, Rpm~1.0-1.07 | See VERIFICATION_LOG item 13 |

## Limitations
- The paper-scale Section 5 runs were done separately (about 2 h in total on 2 cores) and are not part of `run_all.sh`. The amp 0.05 and 0.3 runs use one seed (42); amp 0.1 and 0.2 use four (42, 1, 2, 3); the perturbation is always one localized link; the exponents are finite-time estimates over a growth of roughly one decade at most.
- The scalar-model checks (§2.3, §4.1) test the stated toy models, not the full SU(3) system.
- Results were produced in one environment; no cross-platform comparison was made.

Note (item 12): runs of item 12 use the vectorised engine (agrees with the reference engine to 1e-14 in the fields and <=1.7e-5 relative in the distances, exponents to 4 digits; `results/verify_fast_engine.log`, `results/engine_check/`). Set OMP_NUM_THREADS=1 when running several jobs in parallel (multi-threaded BLAS on 3x3 matrices oversubscribes the cores).


