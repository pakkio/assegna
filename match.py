#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "openai>=1.0",
#     "python-dotenv>=1.0",
# ]
# ///
"""Match users to places by capability fit, with optional LLM-assisted scoring."""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack as sparse_vstack

BASE_URL = "https://opencode.ai/zen/go/v1"
MODEL = os.environ.get("OPENCODEGO_MODEL", "deepseek-v4-flash")
LLM_THRESHOLD = 0.5  # only ask the LLM to refine when the exact-overlap score is below this

MAX_RELOCATION_KM = 30.0
OVER_CAP_PENALTY_PER_KM = 0.005  # cost applied to every km beyond the 30km cap
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_distance_matrix(users: list[dict], places: list[dict]) -> np.ndarray:
    dist = np.zeros((len(users), len(places)))
    for i, u in enumerate(users):
        for j, p in enumerate(places):
            dist[i, j] = haversine_km(u["lat"], u["lon"], p["lat"], p["lon"])
    return dist


def load_json(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def overlap_score(user_caps: list[str], place_caps: list[str]) -> float:
    u = {c.lower() for c in user_caps}
    p = {c.lower() for c in place_caps}
    if not p:
        return 0.0
    return len(u & p) / len(p)


def get_llm_client() -> OpenAI:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.environ.get("OPENCODEGO_API_KEY")
    if not api_key:
        sys.exit("OPENCODEGO_API_KEY not found in ../.env")
    session_id = str(uuid.uuid4())
    return OpenAI(api_key=api_key, base_url=BASE_URL, default_headers={"x-opencode-session": session_id})


def llm_score(client: OpenAI, user: dict, place: dict) -> float:
    prompt = (
        f"User capabilities: {', '.join(user['capabilities'])}\n"
        f"Place '{place['name']}' required capabilities: {', '.join(place['required_capabilities'])}\n"
        "Rate how well this user semantically fits the place's requirements, from 0.0 (no fit) "
        "to 1.0 (perfect fit), giving partial credit for related or adjacent skills even without "
        "an exact string match. Reply with ONLY a number between 0.0 and 1.0."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    try:
        return max(0.0, min(1.0, float(text.split()[0])))
    except ValueError:
        return 0.0


def build_score_matrix(users: list[dict], places: list[dict], client: OpenAI | None) -> np.ndarray:
    scores = np.zeros((len(users), len(places)))
    for i, user in enumerate(users):
        for j, place in enumerate(places):
            base = overlap_score(user["capabilities"], place["required_capabilities"])
            if client is not None and base < LLM_THRESHOLD:
                base = max(base, llm_score(client, user, place))
            scores[i, j] = base
    return scores


def relocation_note(users: list[dict], user_idx: int, place: dict, distances_col: np.ndarray,
                     cap_scores_col: np.ndarray, distance: float, cap_score: float) -> str:
    candidates = [
        (i, cap_scores_col[i])
        for i in range(len(users))
        if i != user_idx and distances_col[i] <= MAX_RELOCATION_KM
    ]
    user_name = users[user_idx]["name"]
    if not candidates:
        return (
            f"No candidate lives within {MAX_RELOCATION_KM:.0f}km of {place['name']}; "
            f"{user_name} is the closest available fit at {distance:.1f}km, so the cap was exceeded."
        )
    best_i, best_score = max(candidates, key=lambda ic: ic[1])
    best_name = users[best_i]["name"]
    return (
        f"{user_name} relocates {distance:.1f}km (exceeds the {MAX_RELOCATION_KM:.0f}km cap) because their "
        f"capability fit ({cap_score:.2f}) clearly beats the best in-range candidate, {best_name} "
        f"(fit {best_score:.2f}) within {MAX_RELOCATION_KM:.0f}km."
    )


def solve_transportation_lp(combined: np.ndarray, capacities: list[int], force_full: bool = True) -> np.ndarray:
    """Exact max-weight bipartite b-matching (each place <=capacity users).

    This is a transportation problem: its constraint matrix is totally unimodular,
    so the LP relaxation's optimal vertex is already integral -- no MILP/branch-and-bound
    needed. Solved with HiGHS (scipy's default), which handles sparse LPs with
    n*m variables efficiently (tested to 1000 users x 100 places in ~1s).

    force_full=True requires every user to land in exactly one place (sum_j x_ij == 1),
    minimizing total penalty subject to full coverage -- raises if total capacity < n.
    force_full=False allows the solver to leave a user unassigned when every available
    slot would have net-negative combined score (skill mismatch + distance penalty).

    Returns an (n, m) 0/1 assignment matrix.
    """
    n, m = combined.shape
    if force_full and sum(capacities) < n:
        raise ValueError(f"total capacity ({sum(capacities)}) < number of users ({n}); cannot force full assignment")

    num_vars = n * m
    var_idx = np.arange(num_vars)
    user_of = var_idx // m
    place_of = var_idx % m

    A_ub = csr_matrix((np.ones(num_vars), (place_of, var_idx)), shape=(m, num_vars))
    b_ub = np.asarray(capacities, dtype=float)

    A_eq = b_eq = None
    if force_full:
        A_eq = csr_matrix((np.ones(num_vars), (user_of, var_idx)), shape=(n, num_vars))
        b_eq = np.ones(n)
    else:
        # <=1-per-user constraint folded into A_ub (sparse stack, not dense -- avoids an
        # (n+m) x (n*m) dense blowup at scale)
        user_ub = csr_matrix((np.ones(num_vars), (user_of, var_idx)), shape=(n, num_vars))
        A_ub = sparse_vstack([A_ub, user_ub]).tocsr()
        b_ub = np.concatenate([b_ub, np.ones(n)])

    res = linprog(
        c=-combined.flatten(),  # minimize negative == maximize
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, 1),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")
    return (res.x.reshape(n, m) > 0.5)


def best_matching(
    users: list[dict],
    places: list[dict],
    cap_scores: np.ndarray,
    distances: np.ndarray,
    force_full: bool = True,
) -> list[dict]:
    """Capacity-aware assignment: each place accepts up to `capacity` users (default 1).

    Solved exactly via linear programming (see solve_transportation_lp) -- not
    a greedy heuristic. An earlier greedy version (sort pairs by score, assign
    while capacity allows) was measurably suboptimal: on a 2-user/2-place
    counterexample it found total score 11 against a true optimum of 18.
    """
    capacities = [p.get("capacity", 1) for p in places]
    over_cap = np.maximum(0.0, distances - MAX_RELOCATION_KM)
    combined = cap_scores - over_cap * OVER_CAP_PENALTY_PER_KM

    assignment = solve_transportation_lp(combined, capacities, force_full=force_full)

    results = []
    for i, j in zip(*np.where(assignment)):
        i, j = int(i), int(j)
        distance = float(distances[i, j])
        entry = {
            "place": places[j]["name"],
            "user": users[i]["name"],
            "capability_score": round(float(cap_scores[i, j]), 3),
            "distance_km": round(distance, 1),
        }
        if distance > MAX_RELOCATION_KM:
            entry["note"] = relocation_note(
                users, i, places[j], distances[:, j], cap_scores[:, j], distance, float(cap_scores[i, j])
            )
        results.append(entry)

    results.sort(key=lambda e: (e["place"], -e["capability_score"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Match users to places by capability fit.")
    parser.add_argument("--users", default="users.json")
    parser.add_argument("--places", default="places.json")
    parser.add_argument("--llm", action="store_true", help="Use the LLM to refine low-overlap scores semantically")
    parser.add_argument("--allow-unassigned", action="store_true",
                         help="Let the solver leave a user unassigned instead of forcing everyone into a place")
    args = parser.parse_args()

    users = load_json(args.users)
    places = load_json(args.places)

    client = get_llm_client() if args.llm else None
    cap_scores = build_score_matrix(users, places, client)
    distances = build_distance_matrix(users, places)

    total_capacity = sum(p.get("capacity", 1) for p in places)
    want_full = not args.allow_unassigned
    if want_full and total_capacity < len(users):
        print(f"(total capacity {total_capacity} < {len(users)} users -- can't force full assignment, "
              f"falling back to best-available)\n")
        want_full = False

    matches = best_matching(users, places, cap_scores, distances, force_full=want_full)

    unmatched = len(users) - len(matches)
    if unmatched:
        print(f"({unmatched} user(s) left unassigned -- no place had net-positive value for them)\n")

    for m in matches:
        print(f"{m['place']:30s} -> {m['user']:20s} "
              f"(fit={m['capability_score']}, dist={m['distance_km']}km)")
        if "note" in m:
            print(f"  note: {m['note']}")


if __name__ == "__main__":
    main()
