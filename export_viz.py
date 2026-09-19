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

    joint_result, home_col, dist = R.solve_joint_ab_c(employees, places, pool, capacity, place_idx)

    assigned_count = np.zeros(n_places)
    moves_by_tier = {"A": [], "B": [], "C": []}
    for i, j, score in joint_result:
        home = int(home_col[i])
        assigned_count[j] += 1
        tier = employees[i]["tier"]
        if j != home:
            moves_by_tier[tier].append({
                "from_id": places[home]["id"], "to_id": places[j]["id"],
                "from": {"lat": places[home]["lat"], "lon": places[home]["lon"], "name": places[home]["name"]},
                "to": {"lat": places[j]["lat"], "lon": places[j]["lon"], "name": places[j]["name"]},
                "distance_km": round(float(dist[i, j]), 1),
                "exceeds_cap": bool(dist[i, j] > R.MAX_RELOCATION_KM),
            })

    def aggregate_routes(moves):
        routes = {}
        for m in moves:
            key = (m["from_id"], m["to_id"])
            r = routes.setdefault(key, {"from": m["from"], "to": m["to"], "count": 0,
                                          "exceeds_count": 0, "distance_km": m["distance_km"]})
            r["count"] += 1
            r["exceeds_count"] += int(m["exceeds_cap"])
        return list(routes.values())

    a_routes = aggregate_routes(moves_by_tier["A"])
    b_routes = aggregate_routes(moves_by_tier["B"])
    c_routes = aggregate_routes(moves_by_tier["C"])

    out = {
        "places": [{"id": p["id"], "name": p["name"], "lat": p["lat"], "lon": p["lon"],
                     "capacity": p["capacity"]} for p in places],
        "a_routes": a_routes,
        "b_routes": b_routes,
        "c_routes": c_routes,
    }
    json.dump(out, open("viz_data.json", "w"), indent=0)
    print(f"places={len(places)} "
          f"a_moves={len(moves_by_tier['A'])}->{len(a_routes)} routes, "
          f"b_moves={len(moves_by_tier['B'])}->{len(b_routes)} routes, "
          f"c_moves={len(moves_by_tier['C'])}->{len(c_routes)} routes")


if __name__ == "__main__":
    main()
