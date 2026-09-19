# assegna

Workforce-to-place matching and reassignment toolkit.

## Files

- `chat.py` -- simple terminal chat against the OpenCode Go LLM gateway.
- `match.py` -- capability-based matching between `users.json` and `places.json`,
  with a 30km relocation cap and exact LP-based optimal assignment.
- `gen_org.py` -- generates synthetic org data: 100 places anchored to real cities
  across 5 continents, ~7600 existing employees (tagged A/B/C mobility tiers, plus
  `protected_category` / `dependents` / `allow_relocating` / `busy` fields, and a
  quantized `capabilities` proficiency level per skill -- see "Quantized skill
  levels" below), 200 new candidate CVs. Every generated point is validated on-land
  (point-in-polygon against real country boundaries) so nothing lands in the sea or
  an empty desert.
- `reassign.py` -- solves the reassignment as one joint linear program (see
  "Strategy" below), with an optional two-stage lexicographic priority pass for
  employees flagged `important` (see "Priority employees" below).
- `export_viz.py` / `gen_map_html.py` / `build_world_paths.py` -- re-run the
  pipeline and render an interactive relocation map (real country outlines,
  proportional arrows) as a self-contained HTML file.
- `bench_haversine.c` / `BenchHaversine.scala` / `bench_haversine.py` -- identical
  dense-distance-matrix benchmark in three languages, isolating how much of
  `reassign.py`'s runtime is "the language" vs. "the algorithm" (see "Language vs.
  algorithm" below).
- `bench_kdtree.c` / `BenchKdtree.scala` / `bench_kdtree.py` -- the same benchmark
  in the same three languages, but using a KD-tree instead of a dense loop -- the
  actual fix now implemented in `reassign.py`'s `combined_score_matrix`.

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

## Quantized skill levels, not binary capability match

`capability_matrix` accepts either a plain list (binary presence -- what
places' `required_capabilities` still use) or a dict `{capability: level}`
with `level` in `0..5` (what employees' and candidates' `capabilities` now
use: 0 inadequate .. 5 genius-level), normalized to `level / 5` as a float.

This is safe to change freely because it only touches the LP's **objective**
vector `c` (the fit scores), never the constraint matrix `A` or its
right-hand side `b`. Total unimodularity -- why the assignment variables land
exactly on 0/1 at the optimum -- is a theorem about `A` and `b` alone; `c` can
be any real number, including these proficiency floats, with zero effect on
integrality. Confirmed on a 30,114-employee / 1,000-place synthetic run: solve
time was unchanged (3.77s) after switching from binary to 6-level capability
scoring.

## Priority ("important") employees: a two-stage lexicographic solve

`solve_joint_priority` (falls back to plain `solve_joint_all` when nobody is
flagged) gives employees with `"important": true` a hard guarantee ordinary
scoring can't: their outcome is *never* worse because of who else needed a
seat, and is completely unaffected by list order.

- **Stage 1** solves just the important group as free movers (exempt from the
  A/B substitution rule, like tier C) with first claim on full capacity.
- **Stage 2** solves everyone else -- remaining employees plus new-hire
  candidates -- jointly, as usual, against whatever capacity is left.

The trade-off: stage 1 can't see stage 2's movers, so a cross-group
substitution (an important A/B's departure backfilled by a not-yet-decided
non-important C) is never considered. In exchange, an important person's
result depends on nobody lower-priority -- a real guarantee, not a score nudge.

This generalizes to a strict per-individual ranking (I1 > I2 > ... > Ik) by
chaining single-person stages instead of one group stage. Measured on 30,114
employees / 1,000 places with 603 important people: ranking them as one group
costs 3.97s; ranking all 603 individually costs 5.09s (603 tiny 1-person
solves at ~2.5ms each, plus the one shared final joint solve) -- barely more,
because each individual stage is a trivial LP over ~9 candidate places. But
be clear about what that buys: full individual ranking reintroduces the exact
order-dependence (a serial dictatorship) this project avoids for the general
population, just scoped to a small, deliberately-prioritized group. For "many
VIPs with a hard ranking," a single-solve big-weight separation (giving VIP
`k` a score weight no combination of lower-ranked people can outbid) gets the
same guarantee without any sequencing at all -- worth it if the chain ever
needs to scale past a small group.

## Scaling to 30,000 employees / 1,000 places

Synthetic benchmark (30,114 employees, 1,000 places, 2,000 new-hire
candidates, `TOP_K=8`, 15-skill pool): **the full joint LP solves in ~2s**
(down from ~3.8-4.1s before the KD-tree fix below). `TOP_K` and skill-pool
size were swept independently up to 100 and 200 respectively without
materially changing solve time -- at this scale HiGHS's per-solve overhead
dominates over edge count, so there's headroom to widen either without a
performance cost; skill-pool size instead affects how many candidates clear
the match bar, not runtime.

Profiling found the LP solve itself is a small fraction of the total (~0.6s
-- confirmed by feeding the identical model to `scipy.linprog` directly and
to a standalone `highs` CLI binary built from source, both landing in the
same 0.6-0.8s range). The dominant cost used to be `combined_score_matrix`'s
dense haversine distance computation over every (employee, place) pair
(~2.4s) -- see "Language vs. algorithm" below for why that was an
algorithmic problem, not a solver or language one, and how it was fixed.

## Language vs. algorithm: the three-language benchmark

`bench_haversine.{c,scala,py}` run the identical dense 30,114 x 1,000-pair
distance computation to answer a natural question once the LP solve turned
out to already be compiled C++ (HiGHS) regardless of caller: would rewriting
the *surrounding* Python in a compiled language meaningfully help?

| implementation | dense loop (all 30.1M pairs) |
|---|---|
| C (`gcc -O3 -march=native`) | ~1.2s |
| Scala/JVM (GraalVM CE 21) | ~1.2-1.4s |
| Python/numpy (vectorized) | ~1.75-2.0s |

C and Scala tie; both beat numpy by a modest ~1.5x -- all three ultimately
call the same `libm` `sin`/`cos`/`asin`, so there isn't much room for one
language to pull ahead.

Compare that to the actual fix: `HARD_CAP_KM = 80` means ~99.8% of those 30M
computed distances get discarded immediately (each employee has on average
only ~2.4 places within reach out of 1,000). `bench_kdtree.{c,scala,py}`
implement the same fix in each language (hand-rolled KD-tree in C and Scala;
`scipy.spatial.cKDTree` in Python, matching what `reassign.py` actually
calls) -- build a tree over places' ECEF positions, then ask each employee
"which places are within `HARD_CAP_KM`?" instead of computing distance to
all 1,000:

| implementation | dense loop | KD-tree | speedup |
|---|---|---|---|
| C | ~1.2s | ~0.01s | ~100-120x |
| Scala/JVM | ~1.2-1.4s | ~0.015-0.02s (after JIT warmup) | ~70-90x |
| Python | ~1.75-2.0s | ~0.03-0.04s | ~50-55x |

All three land within ~2x of each other on the KD-tree version -- the
algorithmic change dwarfs the language choice in both directions: it beats
even the *fastest* dense-loop language by 8-15x, let alone the one it's
directly replacing. All three also agree the fix is correct on the same
grounds: ~2.40-2.43 candidate places per employee on average, independently
computed in each language (different RNGs, so exact matches aren't expected,
but the same distribution confirms the same filtering logic).

This fix is implemented for real in `reassign.py`'s `combined_score_matrix`
(not just benchmarked in isolation) -- exact haversine distance is now only
computed for pairs the KD-tree returns, with every other pair set to a
sentinel value guaranteed beyond every distance cap the rest of the module
checks. Verified to produce byte-identical `combined`/`cap_score`/`dist`
matrices to the old dense version on a held-out test case (max difference:
`0.0`), and measured end to end at 30k employees / 1,000 places:

| | before | after |
|---|---|---|
| `combined_score_matrix` (employees) | ~2.4s | ~0.92s |
| full `solve_joint_all` pipeline | ~3.8-4.1s | ~2.0s |

(The 0.92s is higher than the isolated ~0.03s KD-tree number above because
`combined_score_matrix` also does the capability-fit matmul and a Python-level
loop flattening the ragged per-employee candidate lists into flat arrays for
the whole 30k-person batch -- still a ~2.6x win on that function, ~2x
end-to-end.)

The lesson generalizes past this one function: check whether you're paying
for genuinely necessary computation before reaching for a faster language to
do the same unnecessary work faster.
