#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
# ]
# ///
"""Reassign an existing workforce + place new hires, under a tiered mobility rule:

  A (fixed)   - never moves.
  B (movable) - may relocate to a better-fitting place if it has spare capacity.
  C (backfill)- may move ONLY into a slot vacated by a B move (chain rule), never
                into arbitrary open capacity.
  New hires   - fill whatever capacity is left after B/C moves.

An exact joint solve over ~7600 employees x 100 places with that chain constraint
is a large MILP -- not tractable here. Instead this runs a 3-stage pipeline; each
stage is solved EXACTLY via LP (transportation-problem structure, so the LP
relaxation is already integral), but the stages are sequential, so the overall
result is a documented heuristic, not a proven global optimum.

Each person is only offered their top-K candidate places (by combined score) to
keep every stage's LP small -- also realistic: nobody actually evaluates all 100
places before considering a move.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

MAX_RELOCATION_KM = 30.0
OVER_CAP_PENALTY_PER_KM = 0.06   # a full 0-to-1 fit swing only justifies ~17km over the cap
HARD_CAP_KM = 80.0               # never even offered as a candidate beyond this, regardless of fit
EARTH_RADIUS_KM = 6371.0
TOP_K = 8  # candidate destinations considered per person, besides staying put


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def capability_matrix(entities, key, pool):
    idx = {c: i for i, c in enumerate(pool)}
    m = np.zeros((len(entities), len(pool)))
    for i, e in enumerate(entities):
        for c in e[key]:
            m[i, idx[c]] = 1
    return m


def combined_score_matrix(people, places, pool):
    """(len(people), len(places)) matrix of capability-fit-minus-distance-penalty."""
    p_caps = capability_matrix(people, "capabilities", pool)
    pl_caps = capability_matrix(places, "required_capabilities", pool)
    req_counts = pl_caps.sum(axis=1)
    req_counts[req_counts == 0] = 1
    cap_score = (p_caps @ pl_caps.T) / req_counts  # (people, places)

    p_lat = np.array([p["lat"] for p in people])[:, None]
    p_lon = np.array([p["lon"] for p in people])[:, None]
    pl_lat = np.array([p["lat"] for p in places])[None, :]
    pl_lon = np.array([p["lon"] for p in places])[None, :]
    dist = haversine_km(p_lat, p_lon, pl_lat, pl_lon)

    over_cap = np.maximum(0.0, dist - MAX_RELOCATION_KM)
    combined = cap_score - over_cap * OVER_CAP_PENALTY_PER_KM
    combined = np.where(dist > HARD_CAP_KM, -1e9, combined)  # never offer absurd distances as candidates
    return combined, cap_score, dist


DEPENDENTS_BLOCK_THRESHOLD = 2  # 2+ dependents: no over-cap move offered, regardless of fit


def apply_mobility_rules(combined, dist, people, home_col=None):
    """Mutates nothing; returns a new matrix reflecting each person's individual constraints.

    - allow_relocating == False: only their home place stays viable (full opt-out).
      For candidates (home_col=None), "home" means anywhere within MAX_RELOCATION_KM
      of their own address -- they just won't consider a distant job.
    - protected_category == True, or dependents >= threshold: over-cap (>30km) moves
      are never offered, even though the general HARD_CAP_KM would otherwise allow it.
      They can still move for a better fit as long as it's within the 30km cap.
    """
    combined = combined.copy()
    for i, person in enumerate(people):
        no_consent = not person.get("allow_relocating", True)
        cap_restricted = person.get("protected_category", False) or \
            person.get("dependents", 0) >= DEPENDENTS_BLOCK_THRESHOLD

        if no_consent:
            if home_col is not None:
                mask = np.ones(combined.shape[1], dtype=bool)
                mask[int(home_col[i])] = False
            else:
                mask = dist[i] > MAX_RELOCATION_KM
            combined[i, mask] = -1e9
        elif cap_restricted:
            over_cap_mask = dist[i] > MAX_RELOCATION_KM
            if home_col is not None:
                over_cap_mask = over_cap_mask.copy()
                over_cap_mask[int(home_col[i])] = False  # never block staying home
            combined[i, over_cap_mask] = -1e9
    return combined


def top_k_edges(combined, k, must_include_col=None):
    """For each row, yield (row, col, score) for its top-k columns, plus a forced column if given.

    Columns masked to -1e9 (beyond HARD_CAP_KM) are excluded even if they'd otherwise
    make the top-k, unless forced via must_include_col (used for a person's own home,
    which is always nearby by construction).
    """
    n = combined.shape[0]
    edges = []
    order = np.argsort(-combined, axis=1)[:, :k]
    for i in range(n):
        cols = {c for c in order[i].tolist() if combined[i, c] > -1e8}
        if must_include_col is not None:
            cols.add(int(must_include_col[i]))
        for j in cols:
            edges.append((i, int(j), float(combined[i, j])))
    return edges


def solve_sparse_bmatching(n_entities, edges, place_capacity, force_full=False):
    """Max-weight bipartite b-matching over a SPARSE edge list (not every entity
    connects to every place). Still an exact LP (transportation-problem substructure
    per place, entity-side is a simple <=1/==1 constraint) -- integral at the optimum.

    edges: list of (entity_idx, place_idx, score)
    place_capacity: dict place_idx -> capacity for this stage
    Returns list of (entity_idx, place_idx, score) chosen.
    """
    if not edges:
        return []
    num_vars = len(edges)
    entity_of = np.array([e[0] for e in edges])
    place_of = np.array([e[1] for e in edges])
    scores = np.array([e[2] for e in edges])

    places_used = sorted(set(place_of.tolist()))
    place_row = {p: r for r, p in enumerate(places_used)}
    place_rows = np.array([place_row[p] for p in place_of])

    A_ub_rows = [place_rows]
    A_ub_data = [np.ones(num_vars)]
    b_ub = [place_capacity[p] for p in places_used]

    if not force_full:
        A_ub_rows.append(entity_of + len(places_used))
        A_ub_data.append(np.ones(num_vars))
        b_ub += [1.0] * n_entities
        A_ub = csr_matrix(
            (np.concatenate(A_ub_data), (np.concatenate(A_ub_rows), np.tile(np.arange(num_vars), 2))),
            shape=(len(places_used) + n_entities, num_vars),
        )
        A_eq = b_eq = None
    else:
        A_ub = csr_matrix((A_ub_data[0], (A_ub_rows[0], np.arange(num_vars))), shape=(len(places_used), num_vars))
        b_ub = b_ub
        A_eq = csr_matrix((np.ones(num_vars), (entity_of, np.arange(num_vars))), shape=(n_entities, num_vars))
        b_eq = np.ones(n_entities)

    res = linprog(c=-scores, A_ub=A_ub, b_ub=np.array(b_ub, dtype=float), A_eq=A_eq, b_eq=b_eq,
                   bounds=(0, 1), method="highs")
    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")
    chosen = res.x > 0.5
    return [edges[i] for i in range(num_vars) if chosen[i]]


def main():
    global OVER_CAP_PENALTY_PER_KM, HARD_CAP_KM
    parser = argparse.ArgumentParser()
    parser.add_argument("--penalty", type=float, default=OVER_CAP_PENALTY_PER_KM)
    parser.add_argument("--hardcap", type=float, default=HARD_CAP_KM)
    args = parser.parse_args()
    OVER_CAP_PENALTY_PER_KM = args.penalty
    HARD_CAP_KM = args.hardcap

    places = load_json("org_places.json")
    employees = load_json("org_employees.json")
    candidates = load_json("org_candidates.json")

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

    # ---------- Stage 1: B relocation ----------
    # capacity available to the B pool at place j = capacity_j - A_j - C_j (whatever's left
    # once fixed A and not-yet-moved C are accounted for); shared between B's staying and B's arriving.
    combined, cap_score, dist = combined_score_matrix(b_employees, places, pool)
    home_col = np.array([place_idx[e["place_id"]] for e in b_employees])
    combined = apply_mobility_rules(combined, dist, b_employees, home_col)
    edges = top_k_edges(combined, TOP_K, must_include_col=home_col)

    stage1_budget = capacity - a_count - c_home_count
    stage1_cap = {j: float(stage1_budget[j]) for j in range(n_places)}

    b_result = solve_sparse_bmatching(len(b_employees), edges, stage1_cap, force_full=True)

    b_assigned_count = np.zeros(n_places)
    b_moves, b_stays = [], []
    for i, j, score in b_result:
        home = int(home_col[i])
        b_assigned_count[j] += 1
        if j != home:
            b_moves.append((b_employees[i], places[j], float(dist[i, j]), score))
        else:
            b_stays.append(b_employees[i])

    # net seats actually freed by B churn at each place (departures not refilled by an incoming B)
    b_net_vacancy = np.maximum(0.0, b_home_count - b_assigned_count)

    # ---------- Stage 2: C backfill (only into places with genuine B-churn vacancy) ----------
    combined_c, cap_score_c, dist_c = combined_score_matrix(c_employees, places, pool)
    home_col_c = np.array([place_idx[e["place_id"]] for e in c_employees])
    combined_c = apply_mobility_rules(combined_c, dist_c, c_employees, home_col_c)
    edges_c_all = top_k_edges(combined_c, TOP_K, must_include_col=home_col_c)
    edges_c = [(i, j, s) for (i, j, s) in edges_c_all if b_net_vacancy[j] > 0 or j == int(home_col_c[i])]

    # capacity available to the C pool at place j = the B-churn vacancy (chain rule) plus C's own
    # current seats there (so staying is always an option regardless of b_net_vacancy)
    stage2_budget = b_net_vacancy + c_home_count
    stage2_cap = {j: float(stage2_budget[j]) for j in range(n_places)}

    c_result = solve_sparse_bmatching(len(c_employees), edges_c, stage2_cap, force_full=True)

    c_assigned_count = np.zeros(n_places)
    c_moves, c_stays = [], []
    for i, j, score in c_result:
        home = int(home_col_c[i])
        c_assigned_count[j] += 1
        if j != home:
            c_moves.append((c_employees[i], places[j], float(dist_c[i, j]), score))
        else:
            c_stays.append(c_employees[i])

    # ---------- Stage 3: new hires fill whatever is genuinely left ----------
    occupied_after = a_count + b_assigned_count + c_assigned_count
    final_open = {j: float(max(0.0, capacity[j] - occupied_after[j])) for j in range(n_places)}

    combined_n, cap_score_n, dist_n = combined_score_matrix(candidates, places, pool)
    combined_n = apply_mobility_rules(combined_n, dist_n, candidates, home_col=None)
    edges_n = top_k_edges(combined_n, TOP_K)
    n_result = solve_sparse_bmatching(len(candidates), edges_n, final_open, force_full=False)

    # ---------- report ----------
    print(f"Stage 1 (B relocation): {len(b_moves)} moved / {len(b_stays)} stayed "
          f"(of {len(b_employees)} B employees)")
    exceptions_b = [m for m in b_moves if m[2] > MAX_RELOCATION_KM]
    print(f"  {len(exceptions_b)} exceeded the 30km cap")

    print(f"Stage 2 (C backfill): {len(c_moves)} moved into a B-vacated seat / "
          f"{len(c_stays)} stayed (of {len(c_employees)} C employees)")
    exceptions_c = [m for m in c_moves if m[2] > MAX_RELOCATION_KM]
    print(f"  {len(exceptions_c)} exceeded the 30km cap")

    print(f"Stage 3 (new hires): {len(n_result)} / {len(candidates)} placed")
    unplaced = len(candidates) - len(n_result)
    if unplaced:
        print(f"  {unplaced} candidate(s) left unplaced -- no viable capacity/fit within reach")

    n_assigned_count = np.zeros(n_places)
    for i, j, score in n_result:
        n_assigned_count[j] += 1
    final_occupied = occupied_after + n_assigned_count
    over_capacity = final_occupied > capacity + 1e-6
    print(f"Capacity check: {'OK, no place over capacity' if not over_capacity.any() else f'VIOLATED at {over_capacity.sum()} place(s)'}")

    remaining_vacancy = float((capacity - final_occupied).clip(min=0).sum())
    print(f"Remaining open seats after all moves + hires: {remaining_vacancy:.0f}")

    print("\nSample of exceptional (>30km) relocations:")
    for e in (exceptions_b + exceptions_c)[:5]:
        person, place, d, s = e
        print(f"  {person['name']} ({person['tier']}) -> {place['name']}: {d:.1f}km, fit={s:.2f}")


if __name__ == "__main__":
    main()
