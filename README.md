# DUFT: Numerical verification code

Numerical implementation and verification scripts for the manuscript
*"Dynamic Unified Field Theory (DUFT): A Hamilton–Gradient Dual Framework
with Dissipative Limits, Static Subsets, and Perturbation Growth"*
by Hazime Sendo (ORCID 0009-0007-3285-0308), prepared for submission to
*Chaos* (AIP Publishing).

Repository: https://github.com/Hazime-Sendo/duft-core

## Repository layout

```
src/
  gauge_su3.py                SU(3) link-variable gauge dynamics: staples, force,
                              leapfrog, 4th-order Yoshida, energy, Gauss's law
data/
  parameters.json             Seeds and parameters used by the scripts
                              (no external datasets; all data are generated
                              deterministically from the seed, default 42)
scripts/
  verify_energy_convergence.py        Sec. 3    leapfrog O(dt^2), Yoshida-4 O(dt^4)
  verify_gauss_law.py                 Sec. 6    exact Gauss's-law conservation
  lyapunov_phase_diagram.py           Sec. 5    quick smoke test (16x16, T=5)
  lyapunov_paper_scale.py             Sec. 5    paper-scale (32x32, T=10) finite-time
                              exponents from link AND gauge-invariant plaquette
                              distances, fit-window sensitivity, energy drift;
                              run per amplitude, ~15-45 min each; NOT in run_all.sh
  frw_klein_gordon_derivation.py      Sec. 3 / App. A   2H friction term (sympy)
  verify_adiabatic_elimination.py     Sec. 2.3  exact 2x2 check of adiabatic limit
  verify_static_residual.py           Sec. 4.1  static-solution residual (toy model)
  investigate_symplectic_euler_claim.py  Sec. 3.4 / 6.1 / App. B
                              DOES NOT reproduce the original "10^15-fold,
                              dt-independent blowup" claim; that claim was
                              retracted (see docs/VERIFICATION_LOG.md, item 7)
  analyze_lyapunov_windows.py Sec. 5    fit-window sensitivity from results/lyapunov_amp*.csv
  run_all.sh                  Runs everything and writes results/*.log
results/                      Console output of the last full run (+ environment.txt)
docs/
  VERIFICATION_LOG.md         Sign/normalization errors caught during development
  REPRODUCIBILITY_LOG.md      Per-script outcome of the last full run
requirements.txt              Dependencies
LICENSE                       Apache License 2.0 (code only)
```

## Requirements

Python 3.9+ (tested with 3.11.15). Install with:

```bash
pip install -r requirements.txt     # numpy, scipy, sympy
```

## Running

From the repository root:

```bash
bash scripts/run_all.sh                       # all scripts, ~8 min on one CPU core
python3 scripts/verify_energy_convergence.py  # or any single script
```

Outputs from the last full run are stored in `results/`; their
interpretation against the paper's claims is in
`docs/REPRODUCIBILITY_LOG.md`.

Note on scale: `lyapunov_phase_diagram.py` runs a reduced smoke test
(16x16, T=5). The paper-scale numbers of Section 5 come from
`scripts/lyapunov_paper_scale.py`, run once per amplitude:

```bash
for a in 0.05 0.1 0.2 0.3; do python3 scripts/lyapunov_paper_scale.py --amp $a; done
# extra seeds (as used for amp 0.1 and 0.2):
for s in 1 2 3; do python3 scripts/lyapunov_paper_scale.py --amp 0.1 --seed $s; done
for s in 42 1 2 3; do python3 scripts/lyapunov_paper_scale.py --amp 0.1 --random --seed $s; done   # all-link random perturbation (default seed 42: omit --seed)
for s in 1 2 3 42; do for sh in low mid high; do python3 scripts/lyapunov_paper_scale.py --amp 0.1 --shell $sh --seed $s; done; done   # |k|-shell-restricted perturbation
python3 scripts/analyze_lyapunov_windows.py --summary      # low-k vs high-k per window/seed, local vs random vs shell, paired, amplitude scan
python3 scripts/analyze_lyapunov_windows.py --summary --fine   # finer windows 2-5, 4-7, 6-9, 8-10
python3 scripts/analyze_spectral_ratio.py                  # plaq_lo/plaq_hi versus time per perturbation type

# relaxation tests (fast vectorised engine, default): long time, lattice size, artificial spectra
python3 scripts/verify_fast_engine.py                        # fast vs reference engine
python3 scripts/lyapunov_paper_scale.py --amp 0.1 --random --seed 1 --T 40          # long time
python3 scripts/lyapunov_paper_scale.py --amp 0.1 --random --seed 1 --n 64 --T 40   # lattice size (also --n 128 --T 20)
python3 scripts/lyapunov_paper_scale.py --amp 0.1 --spectrum lowpeak --seed 1 --T 40 # lowpeak|midpeak|highpeak|flat
python3 scripts/lyapunov_paper_scale.py --amp 0.2 --random --seed 1 --T 40 --bg smooth   # backgrounds: white (default) | smooth | lowstrong
python3 scripts/plot_relaxation_backgrounds.py                # figure comparing the three backgrounds
python3 scripts/analyze_relaxation.py                        # per-mode ratio vs time, rates, lattice comparison, figures
```

(output: `results/lyapunov_amp<a>.csv` and `.log`; about 15-45 min per
amplitude on one CPU core). Each run uses one seed and one localized
perturbation, so the exponents are finite-time estimates.

## Scope of the verification log

Items 1-9 of `docs/VERIFICATION_LOG.md` underlie statements in the manuscript.
Items 10-13 are additional exploratory checks (random all-link, shell-targeted
and relaxation tests; background dependence, white-noise background only) that
are **not** part of the manuscript; they are provided for transparency and
reproducibility and make no claim beyond the stated conditions.

## Correctness notes

Formulas in `src/gauge_su3.py` were validated by direct numerical
checks (finite-difference comparison of the force against the
numerically differentiated Wilson action; dt- and lattice-size
independence of the Gauss's-law residual). See `docs/VERIFICATION_LOG.md`
for the incorrect candidate formulas ruled out this way, and for item 7,
a preliminary claim that could not be reproduced and was withdrawn, and
items 8-9 for corrections made after an independent review.

## License

The code (`src/`, `scripts/`) is licensed under the Apache License,
Version 2.0 (see `LICENSE`). The manuscript is licensed separately
(CC BY-SA 4.0) and is not part of this repository.

## Citation

Please cite the manuscript and the Zenodo archive of this repository
(DOI to be added after the first release).
