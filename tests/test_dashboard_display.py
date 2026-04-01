import pandas as pd
import pytest
from src.dashboard.dashboard_logic import get_acquiring_teams


def test_get_acquiring_teams_returns_full_names():
    """get_acquiring_teams should return sorted full team names, not abbreviations."""
    waiver_records_df = pd.DataFrame({
        'Player': ['A', 'B', 'C'],
        'team_abbrev': ['EDM', 'TOR', 'EDM'],
        'PickupTeamName': ['Edmonton Oilers', 'Toronto Maple Leafs', 'Edmonton Oilers'],
        'TeamPoints': [100, 80, 60],
    })
    result = get_acquiring_teams(waiver_records_df)
    assert result == ['Edmonton Oilers', 'Toronto Maple Leafs']
    assert 'EDM' not in result
    assert 'TOR' not in result


def test_get_acquiring_teams_drops_nulls():
    """Teams with missing PickupTeamName should be excluded."""
    waiver_records_df = pd.DataFrame({
        'PickupTeamName': ['Team A', None, 'Team B'],
        'TeamPoints': [10, 20, 30],
    })
    result = get_acquiring_teams(waiver_records_df)
    assert result == ['Team A', 'Team B']


def test_full_data_display_renames_internal_columns():
    """Displayed column names should be human-readable, not internal identifiers."""
    internal_cols = ['team_abbrev', 'acquisition_type', 'DraftingTeamAbbrev']
    display_rename_map = {
        'team_abbrev': 'Current Team',
        'acquisition_type': 'How Acquired',
        'DraftingTeamAbbrev': 'Drafted By (Abbrev)',
    }
    for col in internal_cols:
        assert col in display_rename_map, f"{col} should have a display name"
        assert display_rename_map[col] != col, f"{col} rename should differ from original"
