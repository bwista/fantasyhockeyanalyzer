import os
import tempfile
import pandas as pd
import pytest
from src.dashboard.dashboard_logic import compute_duration_and_avg, process_data, get_data_freshness, get_scoring_categories, get_top_contributors


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


def test_get_scoring_categories_returns_nonzero_keys():
    """Only stat categories with non-zero fantasy point values are returned."""
    stats_df = pd.DataFrame({
        'points_breakdown': [
            {'G': 2.0, 'A': 4.0, 'PPP': 0.0, 'HIT': 0.5},
            {'G': 0.0, 'A': 2.0, 'PPP': 1.0, 'HIT': 0.0},
        ]
    })
    result = get_scoring_categories(stats_df)
    assert result == ['A', 'G', 'HIT', 'PPP']


def test_get_scoring_categories_empty_df():
    """Empty DataFrame returns empty list."""
    stats_df = pd.DataFrame({'points_breakdown': []})
    result = get_scoring_categories(stats_df)
    assert result == []


def test_get_top_contributors_basic():
    """Returns players sorted by total_points descending with aggregated raw stats."""
    stats_df = pd.DataFrame({
        'name': ['Alice', 'Alice', 'Bob', 'Bob'],
        'position': ['Center', 'Center', 'Defense', 'Defense'],
        'team_abbrev': ['ODB', 'ODB', 'ODB', 'ODB'],
        'week': [1, 2, 1, 2],
        'total_points': [10.0, 15.0, 20.0, 5.0],
        'points_breakdown': [
            {'G': 2.0, 'A': 4.0},
            {'G': 4.0, 'A': 2.0},
            {'G': 6.0, 'A': 0.0},
            {'G': 0.0, 'A': 2.0},
        ],
        'stats': [
            {'G': 1.0, 'A': 2.0},
            {'G': 2.0, 'A': 1.0},
            {'G': 3.0, 'A': 0.0},
            {'G': 0.0, 'A': 1.0},
        ],
    })
    scoring_cats = ['A', 'G']
    result = get_top_contributors(stats_df, 'ODB', 1, 2, scoring_cats, top_n=5)
    assert len(result) == 2
    assert set(result['Player'].tolist()) == {'Alice', 'Bob'}
    assert result['TotalPoints'].sum() == pytest.approx(50.0)
    assert 'G' in result.columns
    assert 'A' in result.columns


def test_get_top_contributors_filters_by_team():
    """Only players from the selected team are included."""
    stats_df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'position': ['Center', 'Defense'],
        'team_abbrev': ['ODB', 'JFA'],
        'week': [1, 1],
        'total_points': [10.0, 20.0],
        'points_breakdown': [{'G': 2.0}, {'G': 4.0}],
        'stats': [{'G': 1.0}, {'G': 2.0}],
    })
    result = get_top_contributors(stats_df, 'ODB', 1, 1, ['G'], top_n=5)
    assert len(result) == 1
    assert result.iloc[0]['Player'] == 'Alice'


def test_get_top_contributors_filters_by_week_range():
    """Only weeks within range are included."""
    stats_df = pd.DataFrame({
        'name': ['Alice', 'Alice', 'Alice'],
        'position': ['Center', 'Center', 'Center'],
        'team_abbrev': ['ODB', 'ODB', 'ODB'],
        'week': [1, 2, 3],
        'total_points': [10.0, 20.0, 30.0],
        'points_breakdown': [{'G': 2.0}, {'G': 4.0}, {'G': 6.0}],
        'stats': [{'G': 1.0}, {'G': 2.0}, {'G': 3.0}],
    })
    result = get_top_contributors(stats_df, 'ODB', 1, 2, ['G'], top_n=5)
    assert result.iloc[0]['TotalPoints'] == pytest.approx(30.0)
    assert result.iloc[0]['G'] == pytest.approx(3.0)


def test_get_top_contributors_respects_top_n():
    """Only top N players returned."""
    rows = []
    for i in range(10):
        rows.append({
            'name': f'Player{i}', 'position': 'Center', 'team_abbrev': 'ODB',
            'week': 1, 'total_points': float(i), 'points_breakdown': {'G': 0.0},
            'stats': {'G': 0.0},
        })
    stats_df = pd.DataFrame(rows)
    result = get_top_contributors(stats_df, 'ODB', 1, 1, ['G'], top_n=3)
    assert len(result) == 3
    assert result.iloc[0]['TotalPoints'] == pytest.approx(9.0)


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
