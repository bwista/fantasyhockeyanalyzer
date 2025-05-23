import pytest
import pandas as pd
import numpy as np

# This demonstrates testing for utility functions that could be extracted
# from the dashboard logic for better testability

class TestDataTransformations:
    """Test data transformation utility functions."""
    
    def test_calculate_draft_pick_from_index(self):
        """Test calculating draft pick number from DataFrame index."""
        # Create sample draft data
        draft_data = {
            'Player': ['Player A', 'Player B', 'Player C'],
            'Team': ['Team 1', 'Team 2', 'Team 1']
        }
        df = pd.DataFrame(draft_data)
        
        # Calculate draft pick (1-indexed)
        df['DraftPick'] = df.index + 1
        
        assert df['DraftPick'].tolist() == [1, 2, 3]
        assert df.iloc[0]['DraftPick'] == 1
        assert df.iloc[2]['DraftPick'] == 3
    
    def test_calculate_value_score(self):
        """Test value score calculation (DraftRank - PointsRank)."""
        # Sample data with draft rank and points rank
        test_data = {
            'Player': ['Player A', 'Player B', 'Player C'],
            'DraftRank': [1, 5, 10],
            'PointsRank': [3, 2, 15]
        }
        df = pd.DataFrame(test_data)
        
        # Calculate value score
        df['ValueScore'] = df['DraftRank'] - df['PointsRank']
        
        expected_scores = [-2, 3, -5]  # Negative = better than expected
        assert df['ValueScore'].tolist() == expected_scores
    
    def test_points_rank_calculation(self):
        """Test points ranking (higher points = better rank)."""
        test_data = {
            'Player': ['Player A', 'Player B', 'Player C', 'Player D'],
            'TotalPoints': [100, 150, 75, 150]  # Including tie
        }
        df = pd.DataFrame(test_data)
        
        # Calculate points rank (dense ranking, descending)
        df['PointsRank'] = df['TotalPoints'].rank(method='dense', ascending=False)
        
        # Players B and D tied for 1st (150 pts), A is 2nd (100 pts), C is 3rd (75 pts)
        expected_ranks = [2.0, 1.0, 3.0, 1.0]
        assert df['PointsRank'].tolist() == expected_ranks

class TestAcquisitionTypeLogic:
    """Test acquisition type determination logic."""
    
    def test_determine_drafted_acquisition(self):
        """Test identifying drafted acquisitions."""
        # Sample player-team data
        player_data = {
            'name': ['Player A', 'Player A', 'Player B'],
            'team_abbrev': ['TOR', 'MTL', 'TOR'],
            'FirstWeek': [1, 10, 1],
            'DraftingTeamAbbrev': ['TOR', 'TOR', 'BOS'],
            'DraftPick': [5, 5, np.nan]
        }
        df = pd.DataFrame(player_data)
        
        # Apply logic (simplified version)
        def determine_acquisition_type(row, is_first_record):
            if is_first_record and pd.notna(row['DraftPick']) and row['team_abbrev'] == row['DraftingTeamAbbrev']:
                return 'drafted'
            return 'waiver'
        
        # Mark first records
        df_sorted = df.sort_values(['name', 'FirstWeek'])
        first_records = df_sorted.groupby('name').head(1).index
        
        acquisition_types = []
        for idx, row in df_sorted.iterrows():
            is_first = idx in first_records
            acquisition_types.append(determine_acquisition_type(row, is_first))
        
        df_sorted['acquisition_type'] = acquisition_types
        
        # Player A's first record: drafted by TOR, plays for TOR -> 'drafted'
        # Player A's second record: -> 'waiver' (team change)
        # Player B's first record: drafted by BOS but plays for TOR -> 'waiver'
        expected_types = ['drafted', 'waiver', 'waiver']
        assert df_sorted['acquisition_type'].tolist() == expected_types

class TestDataValidation:
    """Test data validation utilities."""
    
    def test_validate_required_columns(self):
        """Test validation of required DataFrame columns."""
        def validate_required_columns(df, required_cols):
            """Utility function to validate required columns exist."""
            missing_cols = [col for col in required_cols if col not in df.columns]
            return len(missing_cols) == 0, missing_cols
        
        # Test with all required columns present
        df_complete = pd.DataFrame({
            'name': ['Player A'],
            'points': [100],
            'team': ['Team 1']
        })
        
        is_valid, missing = validate_required_columns(df_complete, ['name', 'points', 'team'])
        assert is_valid is True
        assert missing == []
        
        # Test with missing columns
        df_incomplete = pd.DataFrame({
            'name': ['Player A'],
            'points': [100]
        })
        
        is_valid, missing = validate_required_columns(df_incomplete, ['name', 'points', 'team'])
        assert is_valid is False
        assert missing == ['team']
    
    def test_clean_numeric_data(self):
        """Test cleaning and validation of numeric data."""
        test_data = {
            'points': ['100', '50.5', 'invalid', np.nan, 75],
            'games': [82, '70', '', None, 60]
        }
        df = pd.DataFrame(test_data)
        
        # Clean numeric columns
        df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)
        df['games'] = pd.to_numeric(df['games'], errors='coerce').fillna(0)
        
        expected_points = [100.0, 50.5, 0.0, 0.0, 75.0]
        expected_games = [82.0, 70.0, 0.0, 0.0, 60.0]
        
        assert df['points'].tolist() == expected_points
        assert df['games'].tolist() == expected_games

# Fixtures for reusable test data
@pytest.fixture
def sample_draft_data():
    """Sample draft data for testing."""
    return pd.DataFrame({
        'Player': ['Connor McDavid', 'Leon Draisaitl', 'Nathan MacKinnon'],
        'Team': ['Edmonton', 'Calgary', 'Colorado'],
        'Round': [1, 1, 1],
        'Pick': [1, 2, 3]
    })

@pytest.fixture
def sample_stats_data():
    """Sample player stats data for testing."""
    return pd.DataFrame({
        'name': ['Connor McDavid', 'Leon Draisaitl', 'Nathan MacKinnon'],
        'team_abbrev': ['EDM', 'CGY', 'COL'],
        'total_points': [150, 128, 140],
        'week': [1, 1, 1]
    })

@pytest.fixture
def sample_team_mapping():
    """Sample team name to abbreviation mapping."""
    return {
        'Edmonton': 'EDM',
        'Calgary': 'CGY',
        'Colorado': 'COL'
    } 