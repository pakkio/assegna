#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26"]
# ///
"""Dense haversine benchmark: Python/numpy leg of a 3-language (C / Scala / Python)
comparison for reassign.py's actual bottleneck -- combined_score_matrix's distance
computation over every (employee, place) pair, before top-K / HARD_CAP_KM pruning.

Same N_EMP x N_PLACES = 30,114 x 1,000 = 30.1M pairs as the C and Scala versions,
same formula, independently seeded RNGs (not meant to produce matching checksums --
this measures wall-clock, not numerical agreement). Vectorized via numpy broadcasting,
which is the version actually used in reassign.py and actually benchmarked; a naive
Python nested-loop version would be far slower and isn't representative of anything
this codebase does.

Measured on this machine: ~1.75s (vs. ~1.2s in C, ~1.2-1.4s in Scala/JVM -- see
bench_haversine.c and BenchHaversine.scala). The real fix for this bottleneck is
algorithmic (a spatial index / KD-tree, since HARD_CAP_KM discards ~99.8% of these
pairs anyway -- see the KD-tree comparison in reassign.py's git history / README),
not a language rewrite: the ~1.5x gap between numpy and native code here is dwarfed
by the ~50x a spatial index gets you, in any of the three languages.
"""

import time

import numpy as np

N_EMP = 30114
N_PLACES = 1000
EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def main():
    rng = np.random.default_rng(1)
    p_lat = rng.uniform(35.0, 60.0, N_EMP)
    p_lon = rng.uniform(-10.0, 30.0, N_EMP)
    pl_lat = rng.uniform(35.0, 60.0, N_PLACES)
    pl_lon = rng.uniform(-10.0, 30.0, N_PLACES)

    for trial in range(1, 4):
        t0 = time.perf_counter()
        dist = haversine_km(p_lat[:, None], p_lon[:, None], pl_lat[None, :], pl_lon[None, :])
        elapsed = time.perf_counter() - t0
        print(f"Python/numpy dense haversine trial {trial} "
              f"({N_EMP} x {N_PLACES} = {N_EMP * N_PLACES} pairs): {elapsed:.3f}s  "
              f"checksum={dist.flat[12345]:.6f}")


if __name__ == "__main__":
    main()
