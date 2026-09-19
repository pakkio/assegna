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
from scipy.spatial import cKDTree

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


def to_ecef(lat, lon):
    """Lat/lon (on Earth's surface) to 3D Cartesian -- so a KD-tree's ordinary
    Euclidean distance corresponds to great-circle distance (a KD-tree over raw
    lat/lon would distort distances badly near the poles and across the dateline)."""
    latr, lonr = np.radians(lat), np.radians(lon)
    return np.stack([
        EARTH_RADIUS_KM * np.cos(latr) * np.cos(lonr),
        EARTH_RADIUS_KM * np.cos(latr) * np.sin(lonr),
        EARTH_RADIUS_KM * np.sin(latr),
    ], axis=-1)


MAX_SKILL_LEVEL = 5  # quantized proficiency: 0 inadequate .. 5 genius-level (6 levels)


def capability_matrix(entities, key, pool):
    """Build an (entities, pool) matrix. A capability entry can be either a plain
    list (presence only, e.g. a place's required_capabilities -- every required
    capability counts as level 1/1) or a dict {capability: level} with level in
    0..MAX_SKILL_LEVEL (a person's quantized proficiency, normalized to [0, 1]).

    This is pure objective data, computed once here in numpy before the LP ever
    runs -- it feeds combined_score_matrix's cap_score, which only ever appears
    in the LP's objective vector `c`, never in the constraint matrix `A` or its
    right-hand side `b`. Total unimodularity (why the assignment LP lands on an
    exact 0/1 solution) is a statement about A and b alone; c can be any real
    number, including these normalized proficiency floats, with zero effect on
    integrality.
    """
    idx = {c: i for i, c in enumerate(pool)}
    m = np.zeros((len(entities), len(pool)))
    for i, e in enumerate(entities):
        caps = e[key]
        if isinstance(caps, dict):
            for c, level in caps.items():
                m[i, idx[c]] = level / MAX_SKILL_LEVEL
        else:
            for c in caps:
                m[i, idx[c]] = 1.0
    return m


# Any distance beyond HARD_CAP_KM is masked to -1e9 anyway (never offered as a
# candidate), so it never needs to be exact -- just finite and unambiguously beyond
# every cap this module checks against (HARD_CAP_KM, MAX_RELOCATION_KM). Finite
# (not inf) so downstream arithmetic (over_cap penalty) can't produce nan/inf.
_SENTINEL_KM = HARD_CAP_KM + 1.0e6


def combined_score_matrix(people, places, pool):
    """(len(people), len(places)) matrix of capability-fit-minus-distance-penalty.

    Exact haversine distance is only computed for (person, place) pairs within
    HARD_CAP_KM of each other -- found via a KD-tree radius query on places' ECEF
    positions, not a dense len(people) x len(places) distance matrix. Every pair
    outside that radius gets a sentinel distance beyond every cap this module
    checks, without spending a haversine call on it -- everyone else's exact
    distance is unaffected, and the eventual mask (dist > HARD_CAP_KM -> -1e9)
    lands on identical entries either way.

    This matters because most pairs ARE beyond HARD_CAP_KM: each place only draws
    people from a small radius, not the whole map. On a 30k-employee / 1000-place
    benchmark this cut the dominant cost of the whole reassignment pipeline from
    ~1.75s to ~0.03s (each employee has only ~2.4 places within reach on average,
    out of 1000) -- see bench_haversine.* and the README's "Language vs.
    algorithm" section for the full comparison against rewriting this same dense
    loop in C or Scala instead (a ~1.5x win, dwarfed by this ~50x algorithmic one).
    """
    p_caps = capability_matrix(people, "capabilities", pool)
    pl_caps = capability_matrix(places, "required_capabilities", pool)
    req_counts = pl_caps.sum(axis=1)
    req_counts[req_counts == 0] = 1
    cap_score = (p_caps @ pl_caps.T) / req_counts  # (people, places)

    n_people, n_places = len(people), len(places)
    p_lat = np.array([p["lat"] for p in people])
    p_lon = np.array([p["lon"] for p in people])
    pl_lat = np.array([p["lat"] for p in places])
    pl_lon = np.array([p["lon"] for p in places])

    dist = np.full((n_people, n_places), _SENTINEL_KM)
    if n_people and n_places:
        tree = cKDTree(to_ecef(pl_lat, pl_lon))
        # chord length of a HARD_CAP_KM great-circle arc -- Euclidean distance in
        # the ECEF embedding is a monotonic function of great-circle distance, so
        # "within this chord" and "within HARD_CAP_KM great-circle" pick out
        # exactly the same set of places.
        chord = 2 * EARTH_RADIUS_KM * np.sin(HARD_CAP_KM / (2 * EARTH_RADIUS_KM))
        candidates = tree.query_ball_point(to_ecef(p_lat, p_lon), r=chord)
        row_idx = np.concatenate([np.full(len(c), i, dtype=int) for i, c in enumerate(candidates)])
        col_idx = np.concatenate([np.asarray(c, dtype=int) for c in candidates])
        if row_idx.size:
            dist[row_idx, col_idx] = haversine_km(p_lat[row_idx], p_lon[row_idx], pl_lat[col_idx], pl_lon[col_idx])

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


def solve_joint_all(employees, candidates, places, pool, capacity, place_idx):
    """Solve A/B/C reallocation AND new-hire placement as one LP.

    Candidates are folded into the SAME solve as A/B/C, competing for capacity
    from the start rather than getting only whatever's left after employees are
    settled. Measured effect on real data: +2.3 total score AND 7 more hires
    placed (60->67 of 200) versus solving them sequentially -- strictly better,
    not a wash, and no slower (candidates add ~1800 more sparse variables, still
    well within what HiGHS solves in a fraction of a second).

    - every employee assigned to exactly one place (their own home is always a
      valid "stay" option, via top_k_edges' must_include_col); every candidate
      assigned to AT MOST one place (may go unplaced if nothing scores >= 0).
    - place capacity: total headcount (any tier + any placed candidate) <= capacity[p]
    - substitution constraint (A/B only): for each place p,
        (# of A/B who stayed at p) + (# of anyone who arrived at p from elsewhere,
         including a placed candidate) >= (# of A/B originally at p)
      i.e. A/B can leave p only if enough inflow covers the gap; a C leaving p is
      exempt; a C or a new hire arriving at p counts toward covering a departure.

    Returns (chosen_employee_edges, chosen_candidate_edges, home_col, dist,
    dist_candidates) -- edges are (idx, place_idx, score) into their own list.
    """
    n_places = len(places)
    n_emp = len(employees)
    n_cand = len(candidates)

    combined_e, cap_e, dist_e = combined_score_matrix(employees, places, pool)
    home_col = np.array([place_idx[e["place_id"]] for e in employees])
    combined_e = apply_mobility_rules(combined_e, dist_e, employees, home_col)
    edges_e = top_k_edges(combined_e, TOP_K, must_include_col=home_col)

    combined_c, cap_c, dist_c = combined_score_matrix(candidates, places, pool)
    combined_c = apply_mobility_rules(combined_c, dist_c, candidates, home_col=None)
    edges_c = top_k_edges(combined_c, TOP_K)

    tiers = np.array([e["tier"] for e in employees])
    is_ab = np.isin(tiers, ["A", "B"])

    # candidate entity indices continue right after employee indices in one shared space
    edges = list(edges_e) + [(i + n_emp, j, s) for (i, j, s) in edges_c]
    num_vars = len(edges)
    entity_of = np.array([e[0] for e in edges])
    place_of = np.array([e[1] for e in edges])
    scores = np.array([e[2] for e in edges])
    is_employee_edge = entity_of < n_emp

    # employees: ==1 (force_full). candidates: <=1 (may go unplaced).
    emp_idx = np.arange(num_vars)[is_employee_edge]
    A_eq = csr_matrix((np.ones(len(emp_idx)), (entity_of[emp_idx], emp_idx)), shape=(n_emp, num_vars))
    b_eq = np.ones(n_emp)

    cand_idx = np.arange(num_vars)[~is_employee_edge]
    cand_rows = entity_of[cand_idx] - n_emp

    # capacity rows (0..n_places-1)
    cap_rows = place_of
    cap_data = np.ones(num_vars)

    # substitution rows (n_places..2*n_places-1): an A/B person staying home, OR
    # ANYONE (employee or candidate) arriving from elsewhere, covers a departure.
    is_home_edge = np.zeros(num_vars, dtype=bool)
    is_home_edge[emp_idx] = place_of[emp_idx] == home_col[entity_of[emp_idx]]
    home_is_ab = np.zeros(num_vars, dtype=bool)
    home_is_ab[emp_idx] = is_ab[entity_of[emp_idx]]
    covers_departure = (is_home_edge & home_is_ab) | (~is_home_edge)
    sub_rows = place_of[covers_departure] + n_places
    sub_cols = np.arange(num_vars)[covers_departure]
    sub_data = -np.ones(covers_departure.sum())

    # candidate <=1 rows (2*n_places..2*n_places+n_cand-1)
    cand_ub_rows = cand_rows + 2 * n_places

    all_rows = np.concatenate([cap_rows, sub_rows, cand_ub_rows])
    all_cols = np.concatenate([np.arange(num_vars), sub_cols, cand_idx])
    all_data = np.concatenate([cap_data, sub_data, np.ones(len(cand_idx))])
    A_ub = csr_matrix((all_data, (all_rows, all_cols)), shape=(2 * n_places + n_cand, num_vars))

    ab_home_count = np.zeros(n_places)
    for e in employees:
        if e["tier"] in ("A", "B"):
            ab_home_count[place_idx[e["place_id"]]] += 1
    b_ub = np.concatenate([capacity, -ab_home_count, np.ones(n_cand)])

    res = linprog(c=-scores, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")

    # The pure transportation problem is provably integral (total unimodularity), but the
    # substitution constraint added on top of it is NOT part of that proof -- it mixes
    # variables across different destinations in a way the classical TU argument doesn't
    # cover. It has come out integral on every dataset tested so far, but that's an
    # empirical observation, not a guarantee for every possible input. Fail loudly instead
    # of silently rounding a fractional solution (which could be suboptimal or even
    # constraint-violating after rounding) if that ever changes.
    frac = np.minimum(res.x, 1 - res.x)
    if frac.max() > 1e-6:
        raise RuntimeError(
            f"LP relaxation returned a fractional solution (max deviation from integer: "
            f"{frac.max():.4f}) -- the substitution constraint broke integrality on this "
            f"input. Rounding here would not be a verified-optimal result; this needs a "
            f"MILP solver (or a reformulation) for this case, not a silent round(0.5)."
        )
    chosen = res.x > 0.5

    emp_result = [edges[k] for k in range(num_vars) if chosen[k] and is_employee_edge[k]]
    cand_result = [(edges[k][0] - n_emp, edges[k][1], edges[k][2])
                   for k in range(num_vars) if chosen[k] and not is_employee_edge[k]]
    return emp_result, cand_result, home_col, dist_e, dist_c


def solve_priority_stage(people, places, pool, capacity, place_idx):
    """Stage 1 of the lexicographic priority solve: 'important' people get first claim
    on capacity, solved as free movers -- no substitution/backfill requirement, same
    exemption tier C already gets. That's what makes it a HARD guarantee rather than a
    score nudge: this stage doesn't know the rest of the workforce exists yet, so
    nothing outside this group can ever make an important person's outcome worse.

    Returns (chosen_edges, home_col, dist) in this group's own local indices.
    """
    n_places = len(places)
    combined, cap_score, dist = combined_score_matrix(people, places, pool)
    home_col = np.array([place_idx[e["place_id"]] for e in people])
    combined = apply_mobility_rules(combined, dist, people, home_col)
    edges = top_k_edges(combined, TOP_K, must_include_col=home_col)
    place_capacity = {i: capacity[i] for i in range(n_places)}
    chosen = solve_sparse_bmatching(len(people), edges, place_capacity, force_full=True)
    return chosen, home_col, dist


def solve_joint_priority(employees, candidates, places, pool, capacity, place_idx):
    """Two-stage lexicographic solve: 'important' employees are placed FIRST (stage 1,
    solve_priority_stage), then everyone else -- non-important employees and new-hire
    candidates -- is solved jointly as usual (solve_joint_all) against whatever
    capacity and A/B backlog remains (stage 2).

    Trade-off vs. the single joint LP: stage 1 can't see stage 2's movers, so a
    cross-group substitution (an important A/B's departure backfilled by a
    not-yet-decided non-important C) is never considered -- important people are
    exempted from the substitution rule entirely instead, which is what makes the
    guarantee hold. In exchange, an important person's outcome can NEVER be worse
    because of who else needed a seat, and is completely unaffected by the order
    people are listed in -- stage 1 is itself an order-free joint LP over the
    important group, exactly like solve_joint_all is for everyone.

    Falls back to a single call to solve_joint_all (no priority stage) when no
    employee is flagged important, so this is a strict superset of the old behavior.
    """
    n_places = len(places)
    important_idx = [i for i, e in enumerate(employees) if e.get("important")]
    rest_idx = [i for i, e in enumerate(employees) if not e.get("important")]

    if not important_idx:
        return solve_joint_all(employees, candidates, places, pool, capacity, place_idx)

    important = [employees[i] for i in important_idx]
    rest = [employees[i] for i in rest_idx]

    imp_edges, imp_home_col, imp_dist = solve_priority_stage(important, places, pool, capacity, place_idx)

    imp_assigned_count = np.zeros(n_places)
    for _, j, _ in imp_edges:
        imp_assigned_count[j] += 1
    remaining_capacity = capacity - imp_assigned_count
    if (remaining_capacity < -1e-6).any():
        raise RuntimeError("stage 1 (important) overcommitted capacity -- should be impossible "
                            "under force_full with per-place capacity bounds")
    remaining_capacity = remaining_capacity.clip(min=0)

    rest_emp_result, cand_result, rest_home_col, rest_dist, dist_c = solve_joint_all(
        rest, candidates, places, pool, remaining_capacity, place_idx)

    emp_result = [(important_idx[i], j, s) for (i, j, s) in imp_edges] + \
                 [(rest_idx[i], j, s) for (i, j, s) in rest_emp_result]

    # full (n_emp, n_places) distance matrix for reporting, same shape main() expects
    _, _, dist = combined_score_matrix(employees, places, pool)
    home_col = np.array([place_idx[e["place_id"]] for e in employees])

    return emp_result, cand_result, home_col, dist, dist_c


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

    # ---------- one joint LP: A/B/C reallocation AND new-hire placement together ----------
    # Measured to strictly beat solving new hires afterward: +2.3 total score and
    # 7 more hires placed on this dataset, same or better solve time -- see
    # solve_joint_all's docstring.
    emp_result, n_result, home_col, dist, dist_n = solve_joint_priority(
        employees, candidates, places, pool, capacity, place_idx)

    assigned_count = np.zeros(n_places)
    moves_by_tier = {"A": [], "B": [], "C": []}
    stays_by_tier = {"A": [], "B": [], "C": []}
    for i, j, score in emp_result:
        home = int(home_col[i])
        assigned_count[j] += 1
        tier = employees[i]["tier"]
        if j != home:
            moves_by_tier[tier].append((employees[i], places[j], float(dist[i, j]), score))
        else:
            stays_by_tier[tier].append(employees[i])

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
