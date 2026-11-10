"""Compare the vectorised engine (src/gauge_su3_fast.py) with the reference
loops (src/gauge_su3.py): same initial data, 100 leapfrog steps, max abs
difference of links/electric field and of the energy."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.gauge_su3 import generate_background, leapfrog_step, compute_energy
from src.gauge_su3_fast import leapfrog_step_fast, compute_energy_fast

N, steps, dt = 16, 100, 0.01
a = generate_background(N, N, 0.2, np.random.default_rng(7))
b = tuple(x.copy() for x in a)
print("energy at t=0: ref %.15f fast %.15f" % (compute_energy(a[2], a[3], a[0], a[1]), compute_energy_fast(b[2], b[3], b[0], b[1])))
t0 = time.time()
for _ in range(steps): leapfrog_step(*a, dt)
t1 = time.time()
for _ in range(steps): leapfrog_step_fast(*b, dt)
t2 = time.time()
print("max |dU| = %.2e, max |dE| = %.2e" % (max(abs(a[0] - b[0]).max(), abs(a[1] - b[1]).max()),
                                              max(abs(a[2] - b[2]).max(), abs(a[3] - b[3]).max())))
print("energy after %d steps: ref %.14f fast %.14f" % (steps, compute_energy(a[2], a[3], a[0], a[1]),
                                                       compute_energy_fast(b[2], b[3], b[0], b[1])))
print("time ref %.2fs fast %.2fs" % (t1 - t0, t2 - t1))
