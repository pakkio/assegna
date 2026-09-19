#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
# ]
# ///
"""Reassign an existing workforce + place new hires, under a tiered mobility rule:

  A, B (substitution-gated) - may relocate to a better-fitting place, but ONLY if
                someone else (any tier) simultaneously arrives at the place they're
                leaving -- per place, A/B outflow can never exceed any-tier inflow.
  C   (free)  - moves freely, subject only to capacity and the distance rules; never
                needs to arrange a replacement for the seat it vacates.
  New hires   - fill whatever capacity is left after the A/B/C solve.

A and B are solved JOINTLY with C in one LP (not sequential stages): whether an A/B
person is allowed to leave depends on who else arrives at their place in the SAME
solve, which can include a C or another A/B -- so it isn't decomposable into
independent stages the way the old "B first, then C backfills" pipeline was. The
substitution constraint is still linear (each place: sum of "AB stayed or anyone
arrived" edges >= that place's original A/B headcount), so the whole thing remains
one exact LP, not a MILP.

Each person is only offered their top-K candidate places (by combined score) to
keep the LP a manageable size -- also realistic: nobody actually evaluates all 100
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

    - allow_relocating == False, or busy == True: only their home place stays viable
      (full opt-out). "busy" (on a critical project this cycle) is a stronger, temporary
      version of the same block -- distinct from protected_category, which still allows
      in-cap moves. For candidates (home_col=None), "home" means anywhere within
      MAX_RELOCATION_KM of their own address -- they just won't consider a distant job.
    - protected_category == True, or dependents >= threshold: over-cap (>30km) moves
      are never offered, even though the general HARD_CAP_KM would otherwise allow it.
      They can still move for a better fit as long as it's within the 30km cap.
      This applies per-person, not per-tier -- tier A includes "protected" people
      (restricted via protected_category), "important" people (no restriction), and
      "busy" people (fully blocked below), all mixed within the same tier.
    """
    combined = combined.copy()
    for i, person in enumerate(people):
        no_consent = not person.get("allow_relocating", True) or person.get("busy", False)
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


def solve_joint_ab_c(employees, places, pool, capacity, place_idx):
    """Solve A/B/C reallocation as one LP:

    - every employee assigned to exactly one place (their own home is always a
      valid "stay" option, via top_k_edges' must_include_col)
    - place capacity: total headcount (any tier) <= capacity[p]
    - substitution constraint (A/B only): for each place p,
        (# of A/B who stayed at p) + (# of anyone who arrived at p from elsewhere)
        >= (# of A/B originally at p)
      i.e. A/B can leave p only if enough inflow (any tier) covers the gap; a C
      leaving p is exempt (doesn't count as outflow needing coverage), and a C
      arriving at p DOES count toward covering someone else's departure.

    Returns (chosen_edges, home_col, dist) where chosen_edges is a list of
    (employee_idx, place_idx, score).
    """
    n_places = len(places)
    combined, cap_score, dist = combined_score_matrix(employees, places, pool)
    home_col = np.array([place_idx[e["place_id"]] for e in employees])
    combined = apply_mobility_rules(combined, dist, employees, home_col)
    edges = top_k_edges(combined, TOP_K, must_include_col=home_col)

    n = len(employees)
    tiers = np.array([e["tier"] for e in employees])
    is_ab = np.isin(tiers, ["A", "B"])

    num_vars = len(edges)
    entity_of = np.array([e[0] for e in edges])
    place_of = np.array([e[1] for e in edges])
    scores = np.array([e[2] for e in edges])

    # per-person: assigned to exactly one place
    A_eq = csr_matrix((np.ones(num_vars), (entity_of, np.arange(num_vars))), shape=(n, num_vars))
    b_eq = np.ones(n)

    # capacity rows (0..n_places-1): total headcount at p <= capacity[p]
    cap_rows = place_of
    cap_data = np.ones(num_vars)

    # substitution rows (n_places..2*n_places-1): coefficient -1 on any edge that is
    # either (a) an A/B person staying at their own home, or (b) anyone arriving from
    # elsewhere -- both count toward covering that place's A/B departures.
    is_home_edge = place_of == home_col[entity_of]
    home_is_ab = is_ab[entity_of]
    covers_departure = (is_home_edge & home_is_ab) | (~is_home_edge)
    sub_rows = place_of[covers_departure] + n_places
    sub_cols = np.arange(num_vars)[covers_departure]
    sub_data = -np.ones(covers_departure.sum())

    all_rows = np.concatenate([cap_rows, sub_rows])
    all_cols = np.concatenate([np.arange(num_vars), sub_cols])
    all_data = np.concatenate([cap_data, sub_data])
    A_ub = csr_matrix((all_data, (all_rows, all_cols)), shape=(2 * n_places, num_vars))

    ab_home_count = np.zeros(n_places)
    for i, e in enumerate(employees):
        if e["tier"] in ("A", "B"):
            ab_home_count[place_idx[e["place_id"]]] += 1
    b_ub = np.concatenate([capacity, -ab_home_count])

    res = linprog(c=-scores, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")
    chosen = res.x > 0.5
    return [edges[k] for k in range(num_vars) if chosen[k]], home_col, dist


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

    # ---------- Stage 1: joint A/B/C reallocation ----------
    joint_result, home_col, dist = solve_joint_ab_c(employees, places, pool, capacity, place_idx)

    assigned_count = np.zeros(n_places)
    moves_by_tier = {"A": [], "B": [], "C": []}
    stays_by_tier = {"A": [], "B": [], "C": []}
    for i, j, score in joint_result:
        home = int(home_col[i])
        assigned_count[j] += 1
        tier = employees[i]["tier"]
        if j != home:
            moves_by_tier[tier].append((employees[i], places[j], float(dist[i, j]), score))
        else:
            stays_by_tier[tier].append(employees[i])

    # ---------- Stage 2: new hires fill whatever is genuinely left ----------
    final_open = {j: float(max(0.0, capacity[j] - assigned_count[j])) for j in range(n_places)}

    combined_n, cap_score_n, dist_n = combined_score_matrix(candidates, places, pool)
    combined_n = apply_mobility_rules(combined_n, dist_n, candidates, home_col=None)
    edges_n = top_k_edges(combined_n, TOP_K)
    n_result = solve_sparse_bmatching(len(candidates), edges_n, final_open, force_full=False)

    # ---------- report ----------
    for tier, label in [("A", "A relocation"), ("B", "B relocation"), ("C", "C free move")]:
        n_tier = len(moves_by_tier[tier]) + len(stays_by_tier[tier])
        print(f"{label}: {len(moves_by_tier[tier])} moved / {len(stays_by_tier[tier])} stayed (of {n_tier})")
        exceptions = [m for m in moves_by_tier[tier] if m[2] > MAX_RELOCATION_KM]
        print(f"  {len(exceptions)} exceeded the 30km cap")

    print(f"New hires: {len(n_result)} / {len(candidates)} placed")
    unplaced = len(candidates) - len(n_result)
    if unplaced:
        print(f"  {unplaced} candidate(s) left unplaced -- no viable capacity/fit within reach")

    n_assigned_count = np.zeros(n_places)
    for i, j, score in n_result:
        n_assigned_count[j] += 1
    final_occupied = assigned_count + n_assigned_count
    over_capacity = final_occupied > capacity + 1e-6
    print(f"Capacity check: {'OK, no place over capacity' if not over_capacity.any() else f'VIOLATED at {over_capacity.sum()} place(s)'}")

    remaining_vacancy = float((capacity - final_occupied).clip(min=0).sum())
    print(f"Remaining open seats after all moves + hires: {remaining_vacancy:.0f}")

    print("\nSample of exceptional (>30km) relocations:")
    all_exceptions = [m for tier in ("A", "B") for m in moves_by_tier[tier] if m[2] > MAX_RELOCATION_KM]
    for e in all_exceptions[:5]:
        person, place, d, s = e
        print(f"  {person['name']} ({person['tier']}) -> {place['name']}: {d:.1f}km, fit={s:.2f}")


if __name__ == "__main__":
    main()
