#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
# ]
# ///
"""Re-run the reassign.py pipeline and dump places + relocation moves as JSON for charting."""

import json

import numpy as np

import reassign as R


def main():
    places = R.load_json("org_places.json")
    employees = R.load_json("org_employees.json")
    candidates = R.load_json("org_candidates.json")

    pool = sorted({c for p in places for c in p["required_capabilities"]} |
                  {c for e in employees for c in e["capabilities"]} |
                  {c for c_ in candidates for c in c_["capabilities"]})

    n_places = len(places)
    place_idx = {p["id"]: i for i, p in enumerate(places)}
    capacity = np.array([p["capacity"] for p in places], dtype=float)
    a_count = np.zeros(n_places)
    for e in employees:
        if e["tier"] == "A":
            a_count[place_idx[e["place_id"]]] += 1

    b_employees = [e for e in employees if e["tier"] == "B"]
    c_employees = [e for e in employees if e["tier"] == "C"]

    b_home_count = np.zeros(n_places)
    for e in b_employees:
        b_home_count[place_idx[e["place_id"]]] += 1
    c_home_count = np.zeros(n_places)
    for e in c_employees:
        c_home_count[place_idx[e["place_id"]]] += 1

    combined, cap_score, dist = R.combined_score_matrix(b_employees, places, pool)
    home_col = np.array([place_idx[e["place_id"]] for e in b_employees])
    combined = R.apply_mobility_rules(combined, dist, b_employees, home_col)
    edges = R.top_k_edges(combined, R.TOP_K, must_include_col=home_col)
    stage1_budget = capacity - a_count - c_home_count
    stage1_cap = {j: float(stage1_budget[j]) for j in range(n_places)}
    b_result = R.solve_sparse_bmatching(len(b_employees), edges, stage1_cap, force_full=True)

    b_assigned_count = np.zeros(n_places)
    b_moves = []
    for i, j, score in b_result:
        home = int(home_col[i])
        b_assigned_count[j] += 1
        if j != home:
            b_moves.append({
                "from": {"lat": places[home]["lat"], "lon": places[home]["lon"]},
                "to": {"lat": places[j]["lat"], "lon": places[j]["lon"]},
                "distance_km": round(float(dist[i, j]), 1),
                "exceeds_cap": bool(dist[i, j] > R.MAX_RELOCATION_KM),
            })

    b_net_vacancy = np.maximum(0.0, b_home_count - b_assigned_count)

    combined_c, cap_score_c, dist_c = R.combined_score_matrix(c_employees, places, pool)
    home_col_c = np.array([place_idx[e["place_id"]] for e in c_employees])
    combined_c = R.apply_mobility_rules(combined_c, dist_c, c_employees, home_col_c)
    edges_c_all = R.top_k_edges(combined_c, R.TOP_K, must_include_col=home_col_c)
    edges_c = [(i, j, s) for (i, j, s) in edges_c_all if b_net_vacancy[j] > 0 or j == int(home_col_c[i])]
    stage2_budget = b_net_vacancy + c_home_count
    stage2_cap = {j: float(stage2_budget[j]) for j in range(n_places)}
    c_result = R.solve_sparse_bmatching(len(c_employees), edges_c, stage2_cap, force_full=True)

    c_moves = []
    for i, j, score in c_result:
        home = int(home_col_c[i])
        if j != home:
            c_moves.append({
                "from": {"lat": places[home]["lat"], "lon": places[home]["lon"]},
                "to": {"lat": places[j]["lat"], "lon": places[j]["lon"]},
                "distance_km": round(float(dist_c[i, j]), 1),
                "exceeds_cap": bool(dist_c[i, j] > R.MAX_RELOCATION_KM),
            })

    out = {
        "places": [{"id": p["id"], "name": p["name"], "lat": p["lat"], "lon": p["lon"],
                     "capacity": p["capacity"]} for p in places],
        "b_moves": b_moves,
        "c_moves": c_moves,
    }
    json.dump(out, open("viz_data.json", "w"), indent=0)
    print(f"places={len(places)} b_moves={len(b_moves)} c_moves={len(c_moves)}")


if __name__ == "__main__":
    main()
