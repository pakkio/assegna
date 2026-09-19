#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Generate synthetic org data: places (with capacity + 10% starting vacancy),
~8000 existing employees tagged A/B/C (10/70/20%), and 200 new candidate CVs."""

import json
import math
import random

random.seed(7)

CAPS_POOL = [
    "python", "javascript", "react", "sql", "docker", "kubernetes", "terraform", "aws",
    "machine-learning", "pytorch", "go", "rust", "java", "spring", "ux-design", "figma",
    "user-research", "devops", "css", "agile", "project-management", "seo", "marketing",
    "microservices", "systems-programming", "data-analysis", "communication",
]

# Real, named cities as hubs -- not random points in a lat/lon bounding box, which
# routinely landed in the sea (Mediterranean, South Atlantic, mid-Pacific) or in
# uninhabited desert (Sahara, Arabian, Gobi interior). Every hub below is an actual
# populated metro area, so nothing generated around it can end up in open ocean or
# empty desert the way a uniform-random point could.
HUBS = {
    "Africa":        [("Lagos", 6.5244, 3.3792), ("Nairobi", -1.2921, 36.8219),
                       ("Cairo", 30.0444, 31.2357), ("Johannesburg", -26.2041, 28.0473)],
    "Asia":          [("Tokyo", 35.6762, 139.6503), ("Mumbai", 19.0760, 72.8777),
                       ("Beijing", 39.9042, 116.4074), ("Jakarta", -6.2088, 106.8456)],
    "Europe":        [("London", 51.5074, -0.1278), ("Paris", 48.8566, 2.3522),
                       ("Berlin", 52.5200, 13.4050), ("Madrid", 40.4168, -3.7038)],
    "North America": [("New York", 40.7128, -74.0060), ("Mexico City", 19.4326, -99.1332),
                       ("Toronto", 43.6532, -79.3832), ("Chicago", 41.8781, -87.6298)],
    "South America": [("Sao Paulo", -23.5505, -46.6333), ("Buenos Aires", -34.6037, -58.3816),
                       ("Bogota", 4.7110, -74.0721), ("Lima", -12.0464, -77.0428)],
}
CONTINENT_NAMES = list(HUBS.keys())

N_PLACES = 100
NEW_CANDIDATES = 200


def jitter(lat, lon, max_km=15.0):
    # latitude-aware: 1 deg lon = 111km * cos(lat), so this stays roughly isotropic
    # near the equator and near the poles alike, not just at mid-latitudes.
    dlat = random.uniform(-max_km, max_km) / 111.0
    lon_km_per_deg = max(111.0 * abs(math.cos(math.radians(lat))), 5.0)
    dlon = random.uniform(-max_km, max_km) / lon_km_per_deg
    return round(lat + dlat, 4), round(lon + dlon, 4)


def _load_land_rings():
    geo = json.load(open("world.geojson", encoding="utf-8"))
    rings = []
    for feature in geo["features"]:
        geom = feature["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            rings.append(poly[0])  # exterior ring only; holes ignored (negligible at this resolution)
    return rings


LAND_RINGS = _load_land_rings()


def _point_in_ring(lon, lat, ring):
    inside = False
    n = len(ring)
    x1, y1 = ring[0]
    for i in range(1, n + 1):
        x2, y2 = ring[i % n]
        if ((y1 > lat) != (y2 > lat)) and (lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def is_on_land(lat, lon):
    return any(_point_in_ring(lon, lat, ring) for ring in LAND_RINGS)


def jitter_on_land(lat, lon, max_km, max_tries=40):
    """Jitter around (lat, lon) but reject any sample that lands in the sea --
    real hubs are real cities, but a jitter radius can still land offshore."""
    for _ in range(max_tries):
        jlat, jlon = jitter(lat, lon, max_km)
        if is_on_land(jlat, jlon):
            return jlat, jlon
    return round(lat, 4), round(lon, 4)  # fallback: the hub itself is always on land


HUBS_PER_CONTINENT = 4          # metro-area clusters, like Milan/Rome/Naples within Italy
HUB_SPREAD_KM = 40.0            # place jitter around its hub -- comparable to the relocation caps,
                                 # so both in-range and exceptional (>30km, up to ~80km) moves occur

places = []
hubs_by_continent = {name: [(lat, lon) for _, lat, lon in cities] for name, cities in HUBS.items()}
places_per_continent = N_PLACES // len(CONTINENT_NAMES)
places_per_hub = places_per_continent // HUBS_PER_CONTINENT
for ci, continent in enumerate(CONTINENT_NAMES):
    hubs = hubs_by_continent[continent]
    for hi, hub in enumerate(hubs):
        for k in range(places_per_hub):
            i = ci * places_per_continent + hi * places_per_hub + k
            lat, lon = jitter_on_land(hub[0], hub[1], max_km=HUB_SPREAD_KM)
            capacity = random.randint(60, 110)
            vacant = round(capacity * 0.10)
            occupancy = capacity - vacant
            places.append({
                "id": i + 1,
                "name": f"Place{i + 1}",
                "continent": continent,
                "required_capabilities": random.sample(CAPS_POOL, k=3),
                "lat": lat,
                "lon": lon,
                "capacity": capacity,
                "occupancy": occupancy,  # existing A+B+C headcount; capacity - occupancy = starting vacancy
            })

employees = []
emp_id = 1
for place in places:
    occ = place["occupancy"]
    n_a = round(occ * 0.10)
    n_c = round(occ * 0.20)
    n_b = occ - n_a - n_c
    tiers = ["A"] * n_a + ["B"] * n_b + ["C"] * n_c
    for tier in tiers:
        lat, lon = jitter_on_land(place["lat"], place["lon"], max_km=15.0)
        n_caps = random.randint(2, 4)
        employees.append({
            "id": emp_id,
            "name": f"Emp{emp_id}",
            # quantized proficiency per capability: 1 (weakly adapted) .. 5 (genius-level);
            # 0 (inadequate) is never listed -- omission from the dict already means 0
            "capabilities": {c: random.randint(1, 5) for c in random.sample(CAPS_POOL, k=n_caps)},
            "lat": lat,
            "lon": lon,
            "tier": tier,
            "place_id": place["id"],
            "protected_category": random.random() < 0.05,
            "allow_relocating": random.random() < 0.75,
            "dependents": random.choices([0, 1, 2, 3, 4], weights=[40, 30, 15, 10, 5])[0],
            "busy": random.random() < 0.08,  # on a critical project this cycle -- can't move at all, even within cap
        })
        emp_id += 1

candidates = []
for i in range(NEW_CANDIDATES):
    continent = random.choice(CONTINENT_NAMES)
    hub = random.choice(hubs_by_continent[continent])
    lat, lon = jitter_on_land(hub[0], hub[1], max_km=HUB_SPREAD_KM * 1.5)  # candidates spread a bit wider than staff
    candidates.append({
        "id": i + 1,
        "name": f"Candidate{i + 1}",
        "continent": continent,
        "capabilities": {c: random.randint(1, 5) for c in random.sample(CAPS_POOL, k=random.randint(2, 4))},
        "lat": lat,
        "lon": lon,
        "allow_relocating": random.random() < 0.80,
    })

json.dump(places, open("org_places.json", "w"), indent=2)
json.dump(employees, open("org_employees.json", "w"), indent=2)
json.dump(candidates, open("org_candidates.json", "w"), indent=2)

total_capacity = sum(p["capacity"] for p in places)
total_occupancy = sum(p["occupancy"] for p in places)
tiers_count = {"A": 0, "B": 0, "C": 0}
for e in employees:
    tiers_count[e["tier"]] += 1

by_continent = {}
for p in places:
    by_continent[p["continent"]] = by_continent.get(p["continent"], 0) + 1
print(f"places by continent: {by_continent}")
print(f"places: {len(places)}")
print(f"total capacity: {total_capacity}, total occupancy: {total_occupancy} "
      f"(starting vacancy: {total_capacity - total_occupancy})")
print(f"employees: {len(employees)} -> A={tiers_count['A']} B={tiers_count['B']} C={tiers_count['C']}")
print(f"new candidates: {len(candidates)}")
