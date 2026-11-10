#!/usr/bin/env bash
# Run every verification script and store its console output in results/.
# Usage:  bash scripts/run_all.sh        (from the repository root)
set -u
cd "$(dirname "$0")/.."
mkdir -p results
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 --version
  python3 -c "import numpy, scipy, sympy; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| sympy', sympy.__version__)"
  uname -srm
} > results/environment.txt 2>&1
for s in verify_energy_convergence verify_gauss_law lyapunov_phase_diagram \
         frw_klein_gordon_derivation verify_adiabatic_elimination \
         verify_static_residual investigate_symplectic_euler_claim; do
  echo ">>> $s"
  python3 -I "scripts/$s.py" > "results/$s.log" 2>&1
  echo "    exit code: $?" | tee -a "results/exit_codes.txt"
done
