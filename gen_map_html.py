#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Build a self-contained HTML map of relocations from viz_data.json."""

import json

data = json.load(open("viz_data.json"))
places = data["places"]
a_routes = data["a_routes"]
b_routes = data["b_routes"]
c_routes = data["c_routes"]

payload = json.dumps(
    {"places": places, "a_routes": a_routes, "b_routes": b_routes, "c_routes": c_routes},
    separators=(",", ":"),
)
world_path_d = open("world_paths.txt", encoding="utf-8").read()

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
  --series-a:       #1baf7a;
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
    --series-a:       #199e70;
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
  --series-a:       #199e70;
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
.land {{ fill: var(--gridline); stroke: var(--border); stroke-width: 0.4; }}
.place-dot {{ fill: var(--muted); fill-opacity: 0.55; stroke: var(--surface-1); stroke-width: 0.5; }}
.edge-a {{ stroke: var(--series-a); }}
.edge-b {{ stroke: var(--series-b); }}
.edge-c {{ stroke: var(--series-c); }}
.edge {{ fill: none; opacity: 0.6; }}
.edge-a {{ opacity: 0.75; }} /* aqua reads slightly low-contrast on the light surface; compensate */
.edge.exceeds {{ opacity: 0.9; stroke-dasharray: 5 3; }}
.edge:hover {{ opacity: 1; }}
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
  <p class="sub">A/B/C relocations across 100 places, 5 continents. A and B may only move if someone (any tier) backfills their seat; C moves freely. Arrow width = number of people on that route. Dashed = mostly exceeds the 30km cap.</p>
  <div class="stage">
    <div class="stat"><b id="stat-a">-</b> A moves</div>
    <div class="stat"><b id="stat-b">-</b> B moves</div>
    <div class="stat"><b id="stat-c">-</b> C moves</div>
    <div class="stat"><b id="stat-exceed">-</b> exceed 30km cap</div>
  </div>
  <div class="map-wrap">
    <svg id="map" viewBox="0 0 900 450" preserveAspectRatio="xMidYMid meet"></svg>
  </div>
  <div class="legend">
    <div class="legend-item"><span class="swatch" style="background:var(--series-a)"></span> A relocation (substitute required)</div>
    <div class="legend-item"><span class="swatch" style="background:var(--series-b)"></span> B relocation (substitute required)</div>
    <div class="legend-item"><span class="swatch" style="background:var(--series-c)"></span> C free move</div>
    <div class="legend-item"><span class="swatch dashed"></span> mostly exceeds 30km cap</div>
    <div class="legend-item"><span class="dot-swatch"></span> place (size = capacity)</div>
    <div class="legend-item">arrow width = people moved</div>
  </div>
</div>
<div id="tooltip"></div>

<script>
const DATA = {payload};

// standard plate carree world projection: lon in [-180,180], lat in [-90,90], 2:1 aspect
const W = 900, H = 450;
function project(lat, lon) {{
  const x = (lon + 180) / 360 * W;
  const y = (90 - lat) / 180 * H;
  return [x, y];
}}

const svg = document.getElementById("map");
const NS = "http://www.w3.org/2000/svg";
const tooltip = document.getElementById("tooltip");

// real country outlines (Natural-Earth-derived low-res boundaries), baked to this
// same plate-carree projection at build time -- drawn first, everything else layers on top
const land = document.createElementNS(NS, "path");
land.setAttribute("d", "{world_path_d}");
land.setAttribute("class", "land");
svg.appendChild(land);

function showTip(evt, text) {{
  tooltip.textContent = text;
  tooltip.style.left = (evt.clientX + 12) + "px";
  tooltip.style.top = (evt.clientY + 12) + "px";
  tooltip.style.opacity = 1;
}}
function hideTip() {{ tooltip.style.opacity = 0; }}

// Real arrows drawn by hand (shaft + explicit triangular head) rather than SVG
// <marker> refs -- markers render inconsistently small/invisible once markerUnits
// scales off a thin stroke-width inside a viewBox-scaled <svg>. This way the
// arrowhead size is fully controlled and always clearly visible.
function drawRoutes(routes, cls, colorVar, tierLabel) {{
  const maxCount = Math.max(...routes.map(r => r.count), 1);
  const dotShrink = 5; // stop short of the destination dot entirely
  for (const r of routes) {{
    const [x1, y1] = project(r.from.lat, r.from.lon);
    const [x2raw, y2raw] = project(r.to.lat, r.to.lon);
    const dx = x2raw - x1, dy = y2raw - y1;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;       // unit direction vector
    const px = -uy, py = ux;                  // perpendicular unit vector

    const width = 1.2 + 4.5 * Math.sqrt(r.count / maxCount);   // shaft thickness, proportional to count
    const headLen = 5 + width * 1.4;                            // arrowhead length also grows with weight
    const headHalfWidth = 2 + width * 1.1;                      // arrowhead half-width
    const mostlyExceeds = r.exceeds_count / r.count >= 0.5;

    const tipX = x2raw - ux * dotShrink, tipY = y2raw - uy * dotShrink;
    const shaftEndX = tipX - ux * headLen, shaftEndY = tipY - uy * headLen;

    const group = document.createElementNS(NS, "g");
    group.setAttribute("class", "edge " + cls + (mostlyExceeds ? " exceeds" : ""));

    const shaft = document.createElementNS(NS, "line");
    shaft.setAttribute("x1", x1); shaft.setAttribute("y1", y1);
    shaft.setAttribute("x2", shaftEndX); shaft.setAttribute("y2", shaftEndY);
    shaft.setAttribute("stroke-width", width.toFixed(2));
    group.appendChild(shaft);

    const baseX1 = shaftEndX + px * headHalfWidth, baseY1 = shaftEndY + py * headHalfWidth;
    const baseX2 = shaftEndX - px * headHalfWidth, baseY2 = shaftEndY - py * headHalfWidth;
    const head = document.createElementNS(NS, "polygon");
    head.setAttribute("points",
      `${{tipX.toFixed(1)}},${{tipY.toFixed(1)}} ${{baseX1.toFixed(1)}},${{baseY1.toFixed(1)}} ${{baseX2.toFixed(1)}},${{baseY2.toFixed(1)}}`);
    head.setAttribute("fill", colorVar);
    head.setAttribute("stroke", "none");
    group.appendChild(head);

    const exceedsNote = r.exceeds_count > 0 ? `, ${{r.exceeds_count}} exceed the cap` : "";
    group.addEventListener("mousemove", e => showTip(e,
      `${{tierLabel}}: ${{r.from.name}} -> ${{r.to.name}} -- ${{r.count}} people, ~${{r.distance_km}}km${{exceedsNote}}`));
    group.addEventListener("mouseleave", hideTip);
    svg.appendChild(group);
  }}
}}

drawRoutes(DATA.a_routes, "edge-a", "var(--series-a)", "A relocation");
drawRoutes(DATA.b_routes, "edge-b", "var(--series-b)", "B relocation");
drawRoutes(DATA.c_routes, "edge-c", "var(--series-c)", "C free move");

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

const sum = (arr, f) => arr.reduce((s, x) => s + f(x), 0);
document.getElementById("stat-a").textContent = sum(DATA.a_routes, r => r.count);
document.getElementById("stat-b").textContent = sum(DATA.b_routes, r => r.count);
document.getElementById("stat-c").textContent = sum(DATA.c_routes, r => r.count);
document.getElementById("stat-exceed").textContent =
  sum(DATA.a_routes, r => r.exceeds_count) + sum(DATA.b_routes, r => r.exceeds_count) + sum(DATA.c_routes, r => r.exceeds_count);
</script>
"""

open("match_map.html", "w", encoding="utf-8").write(html)
print(f"wrote match_map.html ({len(html)} bytes)")
