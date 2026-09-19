# assegna

Workforce-to-place matching and reassignment toolkit.

- `chat.py` -- simple terminal chat against the OpenCode Go LLM gateway.
- `match.py` -- capability-based matching between `users.json` and `places.json`,
  with a 30km relocation cap and exact LP-based optimal assignment
  (`scipy.optimize.linprog`, transportation-problem formulation).
- `gen_org.py` -- generates synthetic org data (places, existing employees tagged
  A/B/C mobility tiers, new candidate CVs) for the larger reassignment scenario.
- `reassign.py` -- 3-stage reassignment pipeline: B-tier relocation, C-tier chain
  backfill (only into seats vacated by a B move), then new-hire placement into
  whatever capacity remains. Respects per-person mobility constraints
  (`protected_category`, `allow_relocating`, `dependents`).
- `export_viz.py` / `gen_map_html.py` -- re-run the pipeline and render an
  interactive relocation map as a self-contained HTML file.

Run any script with `uv run <script>.py` (each is a self-contained `uv` script
with inline dependencies). `chat.py` and any `--llm` matching paths expect
`OPENCODEGO_API_KEY` in a `.env` file one directory up.
