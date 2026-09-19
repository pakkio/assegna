import pytest
import numpy as np
import reassign
from reassign import (
    haversine_km,
    to_ecef,
    capability_matrix,
    combined_score_matrix,
    apply_mobility_rules,
    top_k_edges,
    solve_sparse_bmatching,
    solve_joint_all,
    solve_priority_stage,
    solve_joint_priority
)
import json
import tempfile
import os
from unittest.mock import patch

def test_haversine_km():
    # Known distance roughly between NYC (40.7128, -74.0060) and LA (34.0522, -118.2437) is ~3935 km
    dist = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900 < dist < 4000

def test_to_ecef():
    # Equator, prime meridian
    ecef = to_ecef(0.0, 0.0)
    assert np.allclose(ecef, [reassign.EARTH_RADIUS_KM, 0.0, 0.0])

def test_capability_matrix():
    pool = ["python", "c", "scala"]
    entities = [
        {"capabilities": {"python": 5, "c": 2}},
        {"capabilities": ["scala", "python"]},
        {"capabilities": {}}
    ]
    mat = capability_matrix(entities, "capabilities", pool)
    assert mat.shape == (3, 3)
    assert np.allclose(mat[0], [1.0, 2/5, 0.0]) # python 5/5, c 2/5, scala 0
    assert np.allclose(mat[1], [1.0, 0.0, 1.0]) # python 1, c 0, scala 1
    assert np.allclose(mat[2], [0.0, 0.0, 0.0])

def test_combined_score_matrix():
    people = [
        {"lat": 0.0, "lon": 0.0, "capabilities": ["python"]},
        {"lat": 1.0, "lon": 1.0, "capabilities": ["python", "c"]}
    ]
    places = [
        {"lat": 0.0, "lon": 0.0, "required_capabilities": ["python"]},
        {"lat": 2.0, "lon": 2.0, "required_capabilities": ["c", "scala"]}
    ]
    pool = ["python", "c", "scala"]
    combined, cap_score, dist = combined_score_matrix(people, places, pool)
    
    assert combined.shape == (2, 2)
    assert cap_score.shape == (2, 2)
    assert dist.shape == (2, 2)
    assert dist[0, 0] < 1.0 # same place
    assert cap_score[0, 0] == 1.0 # exact match

def test_apply_mobility_rules():
    people = [
        # Normal
        {"lat": 0.0, "lon": 0.0, "capabilities": []},
        # No consent
        {"lat": 0.0, "lon": 0.0, "capabilities": [], "allow_relocating": False},
        # Protected (can't move >30km)
        {"lat": 0.0, "lon": 0.0, "capabilities": [], "protected_category": True}
    ]
    combined = np.zeros((3, 3))
    dist = np.array([
        [0.0, 10.0, 50.0],
        [0.0, 10.0, 50.0],
        [0.0, 10.0, 50.0]
    ])
    home_col = np.array([0, 0, 0])
    
    res = apply_mobility_rules(combined, dist, people, home_col)
    
    # Normal can move >30km if combined score is fine, though combined initially is 0
    # but apply_mobility_rules doesn't mask it unless restricted
    assert res[0, 2] == 0.0
    
    # No consent can only stay home (col 0)
    assert res[1, 0] == 0.0
    assert res[1, 1] == -1e9
    assert res[1, 2] == -1e9
    
    # Protected can move <=30km (col 0, 1), not >30km (col 2)
    assert res[2, 0] == 0.0
    assert res[2, 1] == 0.0
    assert res[2, 2] == -1e9

def test_top_k_edges():
    combined = np.array([
        [1.0, 0.5, 0.2, -1e9],
        [-1e9, -1e9, 0.5, 0.8]
    ])
    # Top 2
    edges = top_k_edges(combined, 2, must_include_col=np.array([0, 1]))
    # For row 0: cols 0, 1 are top 2. Must include 0.
    # For row 1: cols 2, 3 are top 2. Must include 1.
    assert len(edges) == 5 # row0: (0,0), (0,1). row1: (1,2), (1,3), plus must include (1,1) even though it's -1e9
    
    assert (0, 0, 1.0) in edges
    assert (0, 1, 0.5) in edges
    assert (1, 3, 0.8) in edges
    assert (1, 2, 0.5) in edges
    assert (1, 1, -1e9) in edges

def test_solve_sparse_bmatching():
    # 2 entities, 2 places
    edges = [
        (0, 0, 1.0),
        (0, 1, 0.5),
        (1, 0, 0.5),
        (1, 1, 1.0)
    ]
    place_capacity = {0: 1, 1: 1}
    chosen = solve_sparse_bmatching(2, edges, place_capacity, force_full=True)
    assert len(chosen) == 2
    assert (0, 0, 1.0) in chosen
    assert (1, 1, 1.0) in chosen

def test_solve_joint_all():
    employees = [
        {"name": "E1", "lat": 0.0, "lon": 0.0, "capabilities": ["python"], "tier": "A", "place_id": "p1"},
        {"name": "E2", "lat": 0.0, "lon": 0.0, "capabilities": ["c"], "tier": "C", "place_id": "p1"}
    ]
    candidates = [
        {"name": "C1", "lat": 0.0, "lon": 0.0, "capabilities": ["python"]}
    ]
    places = [
        {"id": "p1", "name": "P1", "lat": 0.0, "lon": 0.0, "required_capabilities": ["python"], "capacity": 2},
        {"id": "p2", "name": "P2", "lat": 0.0, "lon": 0.0, "required_capabilities": ["c"], "capacity": 2}
    ]
    pool = ["python", "c"]
    place_idx = {"p1": 0, "p2": 1}
    capacity = np.array([2.0, 2.0])
    
    emp_res, cand_res, home_col, dist_e, dist_c = solve_joint_all(
        employees, candidates, places, pool, capacity, place_idx
    )
    
    assert len(emp_res) == 2
    assert len(cand_res) <= 1
    assert len(home_col) == 2

def test_solve_priority_stage():
    people = [
        {"name": "E1", "lat": 0.0, "lon": 0.0, "capabilities": ["python"], "place_id": "p1"}
    ]
    places = [
        {"id": "p1", "name": "P1", "lat": 0.0, "lon": 0.0, "required_capabilities": ["python"], "capacity": 1}
    ]
    pool = ["python"]
    capacity = np.array([1.0])
    place_idx = {"p1": 0}
    chosen, home_col, dist = solve_priority_stage(people, places, pool, capacity, place_idx)
    assert len(chosen) == 1

def test_solve_joint_priority():
    employees = [
        {"name": "E1", "lat": 0.0, "lon": 0.0, "capabilities": ["python"], "tier": "A", "place_id": "p1", "important": True},
        {"name": "E2", "lat": 0.0, "lon": 0.0, "capabilities": ["c"], "tier": "C", "place_id": "p1"}
    ]
    candidates = []
    places = [
        {"id": "p1", "name": "P1", "lat": 0.0, "lon": 0.0, "required_capabilities": ["python"], "capacity": 2},
        {"id": "p2", "name": "P2", "lat": 0.0, "lon": 0.0, "required_capabilities": ["c"], "capacity": 2}
    ]
    pool = ["python", "c"]
    place_idx = {"p1": 0, "p2": 1}
    capacity = np.array([2.0, 2.0])
    
    emp_res, cand_res, home_col, dist_e, dist_c = solve_joint_priority(
        employees, candidates, places, pool, capacity, place_idx
    )
    assert len(emp_res) == 2

@patch("reassign.load_json")
@patch("sys.argv", ["reassign.py"])
def test_main(mock_load_json, capsys):
    employees = [
        {"name": "E1", "lat": 0.0, "lon": 0.0, "capabilities": ["python"], "tier": "A", "place_id": "p1"},
        {"name": "E2", "lat": 0.1, "lon": 0.1, "capabilities": ["c"], "tier": "C", "place_id": "p1"}
    ]
    candidates = [
        {"name": "C1", "lat": 0.0, "lon": 0.0, "capabilities": ["python"]}
    ]
    places = [
        {"id": "p1", "name": "P1", "lat": 0.0, "lon": 0.0, "required_capabilities": ["python"], "capacity": 2},
        {"id": "p2", "name": "P2", "lat": 0.1, "lon": 0.1, "required_capabilities": ["c"], "capacity": 2}
    ]
    
    def side_effect(path):
        if "org_employees.json" in path: return employees
        if "org_candidates.json" in path: return candidates
        if "org_places.json" in path: return places
        return []
    
    mock_load_json.side_effect = side_effect
    
    reassign.main()
    out, err = capsys.readouterr()
    assert "A relocation:" in out
    assert "Capacity check: OK" in out
