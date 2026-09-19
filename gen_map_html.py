#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Build a self-contained HTML map of relocations from viz_data.json."""

import json

data = json.load(open("viz_data.json"))
places = data["places"]
b_moves = data["b_moves"]
c_moves = data["c_moves"]

payload = json.dumps({"places": places, "b_moves": b_moves, "c_moves": c_moves}, separators=(",", ":"))

html = f"""<title>Workforce Relocation Map</title>
<style>
.viz-root {{
  color-scheme: light;
  --surface-1:      #fcfcfb;
  --page:           #f9f9f7;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --muted:          #898781;
  --gridline:       #e1e0d9;
  --series-b:       #2a78d6;
  --series-c:       #eb6834;
  --status-warning: #fab219;
  --border:         rgba(11,11,11,0.10);
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --gridline:       #2c2c2a;
    --series-b:       #3987e5;
    --series-c:       #d95926;
    --status-warning: #fab219;
    --border:         rgba(255,255,255,0.10);
  }}
}}
:root[data-theme="dark"] .viz-root {{
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --page:           #0d0d0d;
  --text-primary:   #ffffff;
  --text-secondary: #c3c2b7;
  --muted:          #898781;
  --gridline:       #2c2c2a;
  --series-b:       #3987e5;
  --series-c:       #d95926;
  --status-warning: #fab219;
  --border:         rgba(255,255,255,0.10);
}}

body {{ margin: 0; background: var(--page); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.viz-root {{ max-width: 980px; margin: 24px auto; padding: 20px 24px 28px; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 12px; color: var(--text-primary); }}
h1 {{ font-size: 18px; margin: 0 0 2px; }}
.sub {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 16px; }}
.stage {{ display: flex; gap: 18px; align-items: center; }}
.stat {{ font-size: 12px; color: var(--text-secondary); }}
.stat b {{ color: var(--text-primary); font-size: 15px; font-variant-numeric: tabular-nums; }}
.map-wrap {{ position: relative; margin-top: 14px; }}
svg {{ width: 100%; height: auto; display: block; background: var(--surface-1); }}
.place-dot {{ fill: var(--muted); fill-opacity: 0.55; stroke: var(--surface-1); stroke-width: 0.5; }}
.edge-b {{ stroke: var(--series-b); }}
.edge-c {{ stroke: var(--series-c); }}
.edge {{ fill: none; stroke-width: 1.1; opacity: 0.55; }}
.edge.exceeds {{ stroke-width: 1.6; opacity: 0.85; stroke-dasharray: 5 3; }}
.edge:hover {{ opacity: 1; stroke-width: 2.4; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 12px; font-size: 12px; color: var(--text-secondary); }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.swatch {{ width: 18px; height: 2px; display: inline-block; }}
.swatch.dashed {{ background: none; border-top: 2px dashed var(--status-warning); width: 18px; height: 0; }}
.dot-swatch {{ width: 9px; height: 9px; border-radius: 50%; background: var(--muted); opacity: 0.55; display: inline-block; }}
#tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1);
  font-size: 11px; padding: 4px 8px; border-radius: 6px; opacity: 0; transition: opacity 0.1s; z-index: 10;
  white-space: nowrap; }}
</style>

<div class="viz-root">
  <h1>Workforce Relocation Map</h1>
  <p class="sub">B-tier relocations and C-tier chain backfills across 100 places (Italy). Dashed = exceeds the 30km cap.</p>
  <div class="stage">
    <div class="stat"><b id="stat-b">-</b> B moves</div>
    <div class="stat"><b id="stat-c">-</b> C backfills</div>
    <div class="stat"><b id="stat-exceed">-</b> exceed 30km cap</div>
  </div>
  <div class="map-wrap">
    <svg id="map" viewBox="0 0 600 712" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
  <div class="legend">
    <div class="legend-item"><span class="swatch" style="background:var(--series-b)"></span> B relocation</div>
    <div class="legend-item"><span class="swatch" style="background:var(--series-c)"></span> C backfill (chain)</div>
    <div class="legend-item"><span class="swatch dashed"></span> exceeds 30km cap</div>
    <div class="legend-item"><span class="dot-swatch"></span> place (size = capacity)</div>
  </div>
</div>
<div id="tooltip"></div>

<script>
const DATA = {payload};

const W = 600, H = 712;
const lats = DATA.places.map(p => p.lat), lons = DATA.places.map(p => p.lon);
const latMin = Math.min(...lats), latMax = Math.max(...lats);
const lonMin = Math.min(...lons), lonMax = Math.max(...lons);
const pad = 24;
const meanLat = (latMin + latMax) / 2 * Math.PI / 180;
const xScale = Math.cos(meanLat);

function project(lat, lon) {{
  const x = pad + (lon - lonMin) * xScale / ((lonMax - lonMin) * xScale) * (W - 2 * pad);
  const y = pad + (latMax - lat) / (latMax - latMin) * (H - 2 * pad);
  return [x, y];
}}

const svg = document.getElementById("map");
const NS = "http://www.w3.org/2000/svg";
const tooltip = document.getElementById("tooltip");

function showTip(evt, text) {{
  tooltip.textContent = text;
  tooltip.style.left = (evt.clientX + 12) + "px";
  tooltip.style.top = (evt.clientY + 12) + "px";
  tooltip.style.opacity = 1;
}}
function hideTip() {{ tooltip.style.opacity = 0; }}

function drawEdges(moves, cls) {{
  for (const m of moves) {{
    const [x1, y1] = project(m.from.lat, m.from.lon);
    const [x2, y2] = project(m.to.lat, m.to.lon);
    const path = document.createElementNS(NS, "line");
    path.setAttribute("x1", x1); path.setAttribute("y1", y1);
    path.setAttribute("x2", x2); path.setAttribute("y2", y2);
    path.setAttribute("class", "edge " + cls + (m.exceeds_cap ? " exceeds" : ""));
    path.addEventListener("mousemove", e => showTip(e, `${{cls === "edge-b" ? "B" : "C"}} move: ${{m.distance_km}}km${{m.exceeds_cap ? " (exceeds cap)" : ""}}`));
    path.addEventListener("mouseleave", hideTip);
    svg.appendChild(path);
  }}
}}

drawEdges(DATA.b_moves, "edge-b");
drawEdges(DATA.c_moves, "edge-c");

const capMax = Math.max(...DATA.places.map(p => p.capacity));
for (const p of DATA.places) {{
  const [x, y] = project(p.lat, p.lon);
  const r = 1.6 + 3.2 * Math.sqrt(p.capacity / capMax);
  const dot = document.createElementNS(NS, "circle");
  dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("r", r);
  dot.setAttribute("class", "place-dot");
  dot.addEventListener("mousemove", e => showTip(e, `${{p.name}} - capacity ${{p.capacity}}`));
  dot.addEventListener("mouseleave", hideTip);
  svg.appendChild(dot);
}}

document.getElementById("stat-b").textContent = DATA.b_moves.length;
document.getElementById("stat-c").textContent = DATA.c_moves.length;
document.getElementById("stat-exceed").textContent =
  DATA.b_moves.filter(m => m.exceeds_cap).length + DATA.c_moves.filter(m => m.exceeds_cap).length;
</script>
"""

open("match_map.html", "w", encoding="utf-8").write(html)
print(f"wrote match_map.html ({len(html)} bytes)")
