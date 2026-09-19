#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Convert world.geojson (Polygon/MultiPolygon, lon/lat) into a single compact SVG
path 'd' string, projected with the same plate-carree projection the map uses
(lon [-180,180] -> x [0,900], lat [-90,90] -> y [0,450])."""

import json

W, H = 900, 450


def project(lon, lat):
    x = (lon + 180) / 360 * W
    y = (90 - lat) / 180 * H
    return x, y


def ring_to_path(ring):
    pts = [project(lon, lat) for lon, lat in ring]
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for x, y in pts[1:]:
        d += f"L{x:.1f},{y:.1f}"
    return d + "Z"


geo = json.load(open("world.geojson", encoding="utf-8"))
parts = []
for feature in geo["features"]:
    geom = feature["geometry"]
    if geom["type"] == "Polygon":
        polygons = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        polygons = geom["coordinates"]
    else:
        continue
    for poly in polygons:
        for ring in poly:
            parts.append(ring_to_path(ring))

d_attr = "".join(parts)
open("world_paths.txt", "w", encoding="utf-8").write(d_attr)
print(f"{len(geo['features'])} countries, {len(parts)} rings, {len(d_attr)} chars")
