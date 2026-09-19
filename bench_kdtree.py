#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.11"]
# ///
"""KD-tree radius-query benchmark: Python leg of the algorithmic-fix comparison that
complements bench_haversine.py. Same 30,114 employees x 1,000 places, but instead of
computing all 30.1M distances, builds a KD-tree over places' ECEF (3D Cartesian)
positions and asks each employee "which places are within HARD_CAP_KM?" -- this uses
scipy.spatial.cKDTree (a library, unlike the hand-rolled trees in bench_kdtree.c /
BenchKdtree.scala) because that's what reassign.py's combined_score_matrix actually
calls; the point of this file is to benchmark the real implementation's approach, not
to hand-roll a third tree for its own sake.

Measured on this machine: ~0.03-0.06s per trial (vs. ~1.75-2.0s for the dense loop in
bench_haversine.py) -- a ~30-50x speedup, consistent with C (~0.01s vs ~1.2s) and
Scala (~0.02s vs ~1.2-1.4s) for the same algorithmic change. See the README's
"Language vs. algorithm" section for the full before/after table.
"""

import time

import numpy as np
from scipy.spatial import cKDTree

N_EMP = 30114
N_PLACES = 1000
EARTH_R_KM = 6371.0
HARD_CAP_KM = 80.0


def to_ecef(lat, lon):
    latr, lonr = np.radians(lat), np.radians(lon)
    return np.stack([
        EARTH_R_KM * np.cos(latr) * np.cos(lonr),
        EARTH_R_KM * np.cos(latr) * np.sin(lonr),
        EARTH_R_KM * np.sin(latr),
    ], axis=-1)


def main():
    rng = np.random.default_rng(1)
    emp_lat = rng.uniform(35.0, 60.0, N_EMP)
    emp_lon = rng.uniform(-10.0, 30.0, N_EMP)
    pl_lat = rng.uniform(35.0, 60.0, N_PLACES)
    pl_lon = rng.uniform(-10.0, 30.0, N_PLACES)

    chord = 2 * EARTH_R_KM * np.sin(HARD_CAP_KM / (2 * EARTH_R_KM))

    for trial in range(1, 4):
        t0 = time.perf_counter()
        tree = cKDTree(to_ecef(pl_lat, pl_lon))
        candidates = tree.query_ball_point(to_ecef(emp_lat, emp_lon), r=chord)
        total_candidates = sum(len(c) for c in candidates)
        elapsed = time.perf_counter() - t0
        print(f"Python/scipy KD-tree trial {trial}: build+query for {N_EMP} employees "
              f"vs {N_PLACES} places: {elapsed:.4f}s  "
              f"avg candidates/employee={total_candidates / N_EMP:.2f}")


if __name__ == "__main__":
    main()
