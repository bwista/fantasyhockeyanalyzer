import os
import tempfile
import pandas as pd
import pytest
from src.dashboard.dashboard_logic import compute_duration_and_avg, process_data, get_data_freshness


def test_duration_includes_both_endpoints():
    """Player on team weeks 1-10 has 10 weeks, not 9."""
    df = pd.DataFrame({
        'Player': ['A'],
        'TeamPoints': [100.0],
        'FirstWeek': [1],
        'LastWeek': [10],
    })
    result = compute_duration_and_avg(df)
    assert result.loc[0, 'Duration'] == 10
    assert result.loc[0, 'AvgPointsPerWeek'] == pytest.approx(10.0)


def test_duration_single_week():
    """Player on team for only 1 week: duration=1, avg=all their points."""
    df = pd.DataFrame({
        'Player': ['B'],
        'TeamPoints': [50.0],
        'FirstWeek': [5],
        'LastWeek': [5],
    })
    result = compute_duration_and_avg(df)
    assert result.loc[0, 'Duration'] == 1
    assert result.loc[0, 'AvgPointsPerWeek'] == pytest.approx(50.0)


def test_avg_points_zero_duration_guard():
    """If FirstWeek == LastWeek + 1 somehow (data error), do not divide by zero."""
    df = pd.DataFrame({
        'Player': ['C'],
        'TeamPoints': [80.0],
        'FirstWeek': [3],
        'LastWeek': [2],  # bad data
    })
    result = compute_duration_and_avg(df)
    assert result.loc[0, 'AvgPointsPerWeek'] == 0.0


def test_get_data_freshness_returns_none_for_missing_file():
    assert get_data_freshness(['/nonexistent/path.json']) is None


def test_get_data_freshness_returns_string_for_existing_file():
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    try:
        result = get_data_freshness([path])
        assert result is not None
        assert 'UTC' in result
    finally:
        os.unlink(path)


def test_process_data_raises_on_bad_stats_schema():
    """process_data should raise ValueError if stats_df is missing required columns."""
    draft_df = pd.DataFrame({
        'Player': ['A'],
        'DraftPick': [1],
        'Team': ['TOR'],
    })
    bad_stats_df = pd.DataFrame({'wrong_col': [1]})  # missing name, team_abbrev, etc.

    with pytest.raises((KeyError, ValueError)):
        process_data(draft_df, bad_stats_df, None)  # positional None for team_map
