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
- **New hires** are folded into the *same* LP too (`solve_joint_all`), competing
  for capacity from the start rather than getting only whatever's left over. A
  placed new hire also counts as a valid "someone arrived" substitute for an
  A/B departure, same as anyone else. Measured against solving them
  afterward: +2.3 total score and 7 more hires placed (60 -> 67 of 200) on the
  same dataset, no slower -- this was tested, not assumed; the sequential
  version was the first cut and got replaced once the joint version proved
  strictly better on every metric.

### Per-person constraints, applied before the solve ever runs

`apply_mobility_rules` masks out infeasible destinations for each person
*before* building the LP (cheaper than adding more constraints to the solver):
- `allow_relocating == False` or `busy == True`: only "stay" is offered.
- `protected_category == True` or `dependents >= 2`: only within the 30km cap
  is offered (no exceptional over-cap moves).
- A distance beyond `HARD_CAP_KM` (80km) is never offered to anyone, tier
  aside -- an over-cap move is only "exceptional" if it's a modest overage
  justified by a real fit gap, not an arbitrarily long one.

### What isn't exact -- and what's only integral so far, not proven integral

The LP finds the true optimum *for the formulation as posed* (top-K candidate
sets, the stated capacity/substitution constraints, the linear
capability-minus-distance-penalty score). The top-K restriction sounded like
an obvious approximation risk, but tested against the real data: `TOP_K=8`
gives the *identical* total score as `TOP_K=100` (every candidate place, no
restriction) -- same solve, same speed. On this dataset the score function
already makes anyone's true best option land within their top-8, so the
restriction currently costs nothing measured, even though nothing guarantees
that in general (a different capability/distance distribution could make it
bite).

More importantly: the pure transportation problem (people <-> places, capacity
only) is *provably* always integral at the LP optimum -- that's the total
unimodularity argument earlier in this doc. The substitution constraint added
on top of it (A/B needs a same-place backfill) is **not** covered by that
proof -- it mixes variables across different destinations in a way the
classical argument doesn't reach. Inspected directly: on the current dataset
the solver's raw output (before any rounding) has **zero** fractional
variables out of ~23,600 -- the LP optimum genuinely is the 0/1 optimum here,
not a lucky rounding. But that's an empirical observation on this instance,
not a general guarantee. `solve_joint_all` now checks this every run and
raises loudly instead of silently rounding if a future dataset ever produces
a fractional solution -- at that point the honest answer to "is this optimal"
would be "not verified," and it would need a MILP solver (or a reformulation
that restores total unimodularity) to actually be sure.

Every constraint the pipeline claims to enforce (capacity limits, the
substitution rule, `busy`/`protected_category` gating) has been independently
re-verified from the raw output at least once during development -- not just
assumed correct because the LP solved successfully. Likewise, "is there a
better solution" was answered by measuring two concrete alternatives (wider
top-K, joint vs. sequential new-hire placement) against the current one,
not by reasoning about it in the abstract.

### A proof that the current solution is optimal -- not just "close enough"

Let `P` be the LP's feasible region (real-valued `x` in `[0, 1]^n` satisfying
the capacity, substitution, and per-person assignment constraints), and let
`I` be the actual integer (0/1) feasible set -- every real assignment of
people to places that respects those same rules. Since integrality is an
*extra* restriction on top of the same linear constraints, `I` subset `P`.

- Let `x*` be the LP's optimum over `P`. For **any** integer-feasible `y` in
  `I`, `y` is also in `P` (dropping the integrality restriction only adds
  more candidates), so `c . y <= c . x*` -- `x*`'s value upper-bounds every
  possible integer assignment, not just ones anyone thought to try.
- If `x*` itself is integral, then `x*` is in `I` too, achieving that bound
  exactly. No `y` in `I` can beat it -- `x*` is optimal over `I`.

That argument is airtight *given* two things are actually true: `x*` really
is the LP's global optimum (not just "the solver reported success"), and
`x*` really is integral (not a rounded fraction). Both were checked directly
rather than assumed:

1. **LP optimality, via an independent duality certificate.** Pulled HiGHS's
   dual solution (shadow prices for every constraint) and computed the dual
   objective separately from the primal. LP weak duality is a theorem: for
   *any* primal-feasible `x` and dual-feasible `y`, `primal(x) >= dual(y)`.
   If they're ever equal, both are proven optimal -- that's the strong
   duality theorem, not solver trust.
   ```
   primal objective: -1384.6884716218524
   dual objective:   -1384.688471621852
   duality gap:       4.5e-13   (floating-point noise -- i.e. zero)
   ```
2. **Integrality.** Max deviation of any variable from `{0, 1}`: `0.0`
   exactly (same check `solve_joint_all` now runs and raises on every call --
   see above).

With both confirmed, the containment argument above is a complete proof, not
an appeal to the solver's authority: **no integer-feasible assignment can
score higher than the one this pipeline returns**, for the model as stated
(this score function, this constraint set, this candidate graph). It does not
claim to be the best possible reassignment under some *other* objective (e.g.
minimizing total distance instead of maximizing fit-minus-penalty) -- that
would be a different LP with a different optimum.
