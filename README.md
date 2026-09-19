# assegna

Workforce-to-place matching and reassignment toolkit.

## Files

- `chat.py` -- simple terminal chat against the OpenCode Go LLM gateway.
- `match.py` -- capability-based matching between `users.json` and `places.json`,
  with a 30km relocation cap and exact LP-based optimal assignment.
- `gen_org.py` -- generates synthetic org data: 100 places anchored to real cities
  across 5 continents, ~7600 existing employees (tagged A/B/C mobility tiers, plus
  `protected_category` / `dependents` / `allow_relocating` / `busy` fields), 200 new
  candidate CVs. Every generated point is validated on-land (point-in-polygon
  against real country boundaries) so nothing lands in the sea or an empty desert.
- `reassign.py` -- solves the reassignment as one joint linear program (see
  "Strategy" below).
- `export_viz.py` / `gen_map_html.py` / `build_world_paths.py` -- re-run the
  pipeline and render an interactive relocation map (real country outlines,
  proportional arrows) as a self-contained HTML file.

Run any script with `uv run <script>.py` (each is a self-contained `uv` script
with inline dependencies). `chat.py` and any `--llm` matching paths expect
`OPENCODEGO_API_KEY` in a `.env` file one directory up.

## Strategy: why this is fast *and* exact, not a heuristic

The short answer for "why does this run in ~2 seconds over 7600 people x 100
places" is: **it's a linear program, not a search.** No trial-and-error, no
greedy pass, no branch-and-bound. `scipy.optimize.linprog` (HiGHS) solves it
directly to a provable optimum in one shot. Two properties make that possible:

### 1. The core problem is a transportation problem, not a general assignment problem

Assigning people to places under capacity limits is a classic **transportation
problem** (bipartite b-matching with capacities). Its constraint matrix is
[totally unimodular](https://en.wikipedia.org/wiki/Unimodular_matrix#Totally_unimodular_matrices),
which means the *linear relaxation* (letting each assignment variable be any
value in `[0, 1]` instead of forcing it to exactly 0 or 1) already has an
integral optimal vertex. So there is no need for a MILP solver with
branch-and-bound -- plain LP already gives a 0/1 answer.

This is why we deliberately do **not** use a greedy heuristic (sort by score,
assign what's left). Early in this project a greedy version was tried and
measurably wrong: a 2-person/2-place counterexample found total score 11
against a true optimum of 18. The LP formulation doesn't have that failure
mode -- it's exact by construction, not "usually good enough."

### 2. Every person only sees their top-K candidate places, not all 100

Nobody realistically evaluates every place in the world before considering a
move. Each person's LP variables are restricted to their top-K
(`TOP_K = 8`) best-scoring destinations plus their own current place (always
offered, so "stay" is always feasible). This keeps the LP sparse: instead of
~7600 x 100 = 760,000 variables, there are on the order of 7600 x 9 ≈ 68,000 --
small enough that HiGHS solves it in about a second, and it doubles as a
realism constraint (bounded relocation search radius), not just a performance
trick.

### The mobility rule, and why it's still one LP and not a MILP

- **A, B** (substitution-gated): may relocate to a better-fitting place, but
  only if someone else -- any tier -- simultaneously arrives at the place
  they're leaving. Per place: `(A/B who stayed) + (anyone who arrived from
  elsewhere) >= (A/B originally there)`.
- **C** (free): moves under the same capacity/distance rules, but never needs
  a replacement to leave.
- Both A/B and C are solved **together in one LP** (`solve_joint_ab_c`), not
  as separate sequential stages -- whether an A/B person can leave depends on
  who else arrives in the *same* solve, which can include a C. That dependency
  is still just one linear inequality per place (100 extra constraint rows),
  so the whole thing remains a single LP, not a harder combinatorial problem.
- **New hires** are placed afterward, filling whatever capacity is left. They
  don't affect anyone else's decision, so they don't need to be in the joint
  solve.

### Per-person constraints, applied before the solve ever runs

`apply_mobility_rules` masks out infeasible destinations for each person
*before* building the LP (cheaper than adding more constraints to the solver):
- `allow_relocating == False` or `busy == True`: only "stay" is offered.
- `protected_category == True` or `dependents >= 2`: only within the 30km cap
  is offered (no exceptional over-cap moves).
- A distance beyond `HARD_CAP_KM` (80km) is never offered to anyone, tier
  aside -- an over-cap move is only "exceptional" if it's a modest overage
  justified by a real fit gap, not an arbitrarily long one.

### What isn't exact

The LP finds the true optimum *for the formulation as posed* (top-K candidate
sets, the stated capacity/substitution constraints, the linear
capability-minus-distance-penalty score). Restricting to top-K candidates is
itself an approximation of "consider every possible move" -- a person's true
best option could in principle sit outside their top-8, though in practice the
score function makes that unlikely. New-hire placement, run after the A/B/C
solve, is sequential rather than jointly optimized with it.

Every constraint the pipeline claims to enforce (capacity limits, the
substitution rule, `busy`/`protected_category` gating) has been independently
re-verified from the raw output at least once during development -- not just
assumed correct because the LP solved successfully.
