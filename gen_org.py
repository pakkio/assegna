#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Generate synthetic org data: places (with capacity + 10% starting vacancy),
~8000 existing employees tagged A/B/C (10/70/20%), and 200 new candidate CVs."""

import json
import random

random.seed(7)

CAPS_POOL = [
    "python", "javascript", "react", "sql", "docker", "kubernetes", "terraform", "aws",
    "machine-learning", "pytorch", "go", "rust", "java", "spring", "ux-design", "figma",
    "user-research", "devops", "css", "agile", "project-management", "seo", "marketing",
    "microservices", "systems-programming", "data-analysis", "communication",
]

LAT_RANGE = (36.6, 47.1)  # Italy bounding box
LON_RANGE = (6.6, 18.5)

N_PLACES = 100
NEW_CANDIDATES = 200


def random_point():
    return round(random.uniform(*LAT_RANGE), 4), round(random.uniform(*LON_RANGE), 4)


def jitter(lat, lon, max_km=15.0):
    # cheap local jitter: ~1 deg lat = 111km
    dlat = random.uniform(-max_km, max_km) / 111.0
    dlon = random.uniform(-max_km, max_km) / (111.0 * 0.75)
    return round(lat + dlat, 4), round(lon + dlon, 4)


places = []
for i in range(N_PLACES):
    lat, lon = random_point()
    capacity = random.randint(60, 110)
    vacant = round(capacity * 0.10)
    occupancy = capacity - vacant
    places.append({
        "id": i + 1,
        "name": f"Place{i + 1}",
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
        lat, lon = jitter(place["lat"], place["lon"])
        employees.append({
            "id": emp_id,
            "name": f"Emp{emp_id}",
            "capabilities": random.sample(CAPS_POOL, k=random.randint(2, 4)),
            "lat": lat,
            "lon": lon,
            "tier": tier,
            "place_id": place["id"],
            "protected_category": random.random() < 0.05,
            "allow_relocating": random.random() < 0.75,
            "dependents": random.choices([0, 1, 2, 3, 4], weights=[40, 30, 15, 10, 5])[0],
        })
        emp_id += 1

candidates = []
for i in range(NEW_CANDIDATES):
    lat, lon = random_point()
    candidates.append({
        "id": i + 1,
        "name": f"Candidate{i + 1}",
        "capabilities": random.sample(CAPS_POOL, k=random.randint(2, 4)),
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

print(f"places: {len(places)}")
print(f"total capacity: {total_capacity}, total occupancy: {total_occupancy} "
      f"(starting vacancy: {total_capacity - total_occupancy})")
print(f"employees: {len(employees)} -> A={tiers_count['A']} B={tiers_count['B']} C={tiers_count['C']}")
print(f"new candidates: {len(candidates)}")
