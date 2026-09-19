import pytest
import numpy as np
import match
from match import (
    haversine_km,
    build_distance_matrix,
    overlap_score,
    build_score_matrix,
    relocation_note,
    solve_transportation_lp,
    best_matching,
    load_json,
    get_llm_client,
    llm_score
)
from unittest.mock import patch, MagicMock
import os
import json
import tempfile

def test_haversine_km():
    dist = haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900 < dist < 4000

def test_build_distance_matrix():
    users = [{"lat": 0.0, "lon": 0.0}, {"lat": 1.0, "lon": 1.0}]
    places = [{"lat": 0.0, "lon": 0.0}]
    dist = build_distance_matrix(users, places)
    assert dist.shape == (2, 1)

def test_overlap_score():
    assert overlap_score(["python", "c"], ["python"]) == 1.0
    assert overlap_score(["python"], ["python", "c"]) == 0.5
    assert overlap_score([], ["python"]) == 0.0
    assert overlap_score(["java"], []) == 0.0

@patch("match.llm_score")
def test_build_score_matrix(mock_llm_score):
    users = [{"capabilities": ["python"]}]
    places = [{"required_capabilities": ["c"]}]
    
    mock_llm_score.return_value = 0.8
    client = MagicMock()
    
    scores = build_score_matrix(users, places, client)
    assert scores.shape == (1, 1)
    assert scores[0, 0] == 0.8
    
    scores_no_llm = build_score_matrix(users, places, None)
    assert scores_no_llm[0, 0] == 0.0

def test_relocation_note():
    users = [{"name": "A"}, {"name": "B"}]
    place = {"name": "P"}
    distances_col = np.array([50.0, 10.0])
    cap_scores_col = np.array([1.0, 0.4])
    
    note = relocation_note(users, 0, place, distances_col, cap_scores_col, 50.0, 1.0)
    assert "exceeds" in note

def test_solve_transportation_lp():
    combined = np.array([
        [1.0, -10.0],
        [-10.0, 1.0]
    ])
    capacities = [1, 1]
    assignment = solve_transportation_lp(combined, capacities, force_full=True)
    assert assignment.shape == (2, 2)
    
    # Test force_full=False
    assignment_not_full = solve_transportation_lp(combined, capacities, force_full=False)
    assert assignment_not_full.shape == (2, 2)

    # Test error when capacity < users and force_full=True
    with pytest.raises(ValueError):
        solve_transportation_lp(combined, [1, 0], force_full=True)
        
    # Mock linprog failure
    with patch("match.linprog") as mock_linprog:
        mock_linprog.return_value = MagicMock(success=False, message="failed")
        with pytest.raises(RuntimeError):
            solve_transportation_lp(combined, capacities, force_full=False)

def test_best_matching():
    users = [{"name": "U1"}, {"name": "U2"}]
    places = [{"name": "P1", "capacity": 1}, {"name": "P2", "capacity": 1}]
    cap_scores = np.array([[1.0, 0.0], [0.0, 1.0]])
    distances = np.array([[0.0, 100.0], [100.0, 0.0]])
    
    matches = best_matching(users, places, cap_scores, distances, force_full=True)
    assert len(matches) == 2
    
    # Check relocation note branch
    users2 = [{"name": "U1"}]
    places2 = [{"name": "P1", "capacity": 1}]
    cap_scores2 = np.array([[1.0]])
    distances2 = np.array([[50.0]])
    matches2 = best_matching(users2, places2, cap_scores2, distances2, force_full=True)
    assert "note" in matches2[0]

def test_load_json():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write('[{"test": 1}]')
        f.close()
        res = load_json(f.name)
        assert len(res) == 1
        os.unlink(f.name)

@patch("match.os.environ.get")
@patch("match.OpenAI")
def test_get_llm_client(mock_openai, mock_env_get):
    mock_env_get.side_effect = lambda k, default=None: "key" if k == "OPENCODEGO_API_KEY" else default
    client = get_llm_client()
    assert client is not None
    
    mock_env_get.side_effect = lambda k, default=None: "" if k == "OPENCODEGO_API_KEY" else default
    with pytest.raises(SystemExit):
        get_llm_client()

def test_llm_score():
    client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "0.9"
    client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    
    score = llm_score(client, {"capabilities": ["python"]}, {"name": "P1", "required_capabilities": ["python"]})
    assert score == 0.9
    
    # Test ValueError branch
    mock_choice.message.content = "not a float"
    score2 = llm_score(client, {"capabilities": ["python"]}, {"name": "P1", "required_capabilities": ["python"]})
    assert score2 == 0.0

@patch("match.get_llm_client")
@patch("match.load_json")
@patch("sys.argv", ["match.py", "--llm", "--allow-unassigned"])
def test_main(mock_load_json, mock_get_llm, capsys):
    mock_get_llm.return_value = MagicMock()
    users = [
        {"name": "U1", "lat": 0.0, "lon": 0.0, "capabilities": ["python"]},
        {"name": "U2", "lat": 0.0, "lon": 0.0, "capabilities": ["c"]}
    ]
    places = [{"name": "P1", "lat": 0.0, "lon": 0.0, "required_capabilities": ["python"], "capacity": 1}]
    
    def side_effect(path):
        if "users" in path: return users
        if "places" in path: return places
        return []
    
    mock_load_json.side_effect = side_effect
    
    match.main()
    out, err = capsys.readouterr()
    assert "U1" in out
    
    # Now test when total_capacity < len(users) and want_full=True
    with patch("sys.argv", ["match.py"]):
        match.main()
        out2, err2 = capsys.readouterr()
        assert "falling back to best-available" in out2
        assert "user(s) left unassigned" in out2

