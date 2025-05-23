import pytest
import pandas as pd
from unittest.mock import Mock, patch
import json

from src.data_processing.parse_draft_results import parse_draft_results

class TestParseDraftResults:
    """Test the parse_draft_results function."""
    
    def test_successful_draft_parsing(self):
        """Test successful parsing of draft results."""
        # Mock the ESPN League and draft picks
        mock_league = Mock()
        mock_pick1 = Mock()
        mock_pick1.__str__ = Mock(return_value="R:1 P:1, Connor McDavid, Team(Edmonton)")
        mock_pick2 = Mock()
        mock_pick2.__str__ = Mock(return_value="R:1 P:2, Leon Draisaitl, Team(Calgary)")
        mock_pick3 = Mock()
        mock_pick3.__str__ = Mock(return_value="R:2 P:15, Nathan MacKinnon, Team(Edmonton)")
        
        mock_league.draft = [mock_pick1, mock_pick2, mock_pick3]
        
        with patch('src.data_processing.parse_draft_results.League', return_value=mock_league):
            result = parse_draft_results(
                league_id=12345,
                year=2023,
                swid="test_swid",
                espn_s2="test_espn_s2"
            )
        
        # Verify the result is a DataFrame with correct structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ['Round', 'Pick', 'Team', 'Player']
        
        # Verify the data was parsed correctly
        assert result.iloc[0]['Round'] == 1
        assert result.iloc[0]['Pick'] == 1
        assert result.iloc[0]['Team'] == 'Edmonton'
        assert result.iloc[0]['Player'] == 'Connor McDavid'
        
        assert result.iloc[1]['Round'] == 1
        assert result.iloc[1]['Pick'] == 2
        assert result.iloc[1]['Team'] == 'Calgary'
        assert result.iloc[1]['Player'] == 'Leon Draisaitl'
        
        assert result.iloc[2]['Round'] == 2
        assert result.iloc[2]['Pick'] == 15
        assert result.iloc[2]['Team'] == 'Edmonton'
        assert result.iloc[2]['Player'] == 'Nathan MacKinnon'
    
    def test_league_connection_failure(self):
        """Test handling of league connection failure."""
        with patch('src.data_processing.parse_draft_results.League', side_effect=Exception("Connection failed")):
            result = parse_draft_results(
                league_id=12345,
                year=2023,
                swid="test_swid",
                espn_s2="test_espn_s2"
            )
        
        assert result is None
    
    def test_draft_fetch_failure(self):
        """Test handling of draft fetch failure."""
        mock_league = Mock()
        mock_league.draft = Mock(side_effect=Exception("Draft fetch failed"))
        
        with patch('src.data_processing.parse_draft_results.League', return_value=mock_league):
            result = parse_draft_results(
                league_id=12345,
                year=2023,
                swid="test_swid",
                espn_s2="test_espn_s2"
            )
        
        assert result is None
    
    def test_empty_draft_results(self):
        """Test handling of empty draft results."""
        mock_league = Mock()
        mock_league.draft = []
        
        with patch('src.data_processing.parse_draft_results.League', return_value=mock_league):
            result = parse_draft_results(
                league_id=12345,
                year=2023,
                swid="test_swid",
                espn_s2="test_espn_s2"
            )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ['Round', 'Pick', 'Team', 'Player']

class TestDraftStringParsing:
    """Test the draft string parsing logic with edge cases."""
    
    def test_different_pick_formats(self):
        """Test parsing of different pick string formats."""
        mock_league = Mock()
        
        # Test various formats that might come from ESPN API
        mock_pick1 = Mock()
        mock_pick1.__str__ = Mock(return_value="R:10 P:150, Player Name-Hyphen, Team(Team Name With Spaces)")
        mock_pick2 = Mock()
        mock_pick2.__str__ = Mock(return_value="R:5 P:67, O'Player With Apostrophe, Team(Short)")
        
        mock_league.draft = [mock_pick1, mock_pick2]
        
        with patch('src.data_processing.parse_draft_results.League', return_value=mock_league):
            result = parse_draft_results(
                league_id=12345,
                year=2023,
                swid="test_swid",
                espn_s2="test_espn_s2"
            )
        
        # Verify edge cases are handled correctly
        assert result.iloc[0]['Round'] == 10
        assert result.iloc[0]['Pick'] == 150
        assert result.iloc[0]['Team'] == 'Team Name With Spaces'
        assert result.iloc[0]['Player'] == 'Player Name-Hyphen'
        
        assert result.iloc[1]['Round'] == 5
        assert result.iloc[1]['Pick'] == 67
        assert result.iloc[1]['Team'] == 'Short'
        assert result.iloc[1]['Player'] == "O'Player With Apostrophe"

# Fixtures for test data
@pytest.fixture
def sample_draft_picks():
    """Sample draft picks for testing."""
    picks = []
    for i in range(3):
        pick = Mock()
        pick.__str__ = Mock(return_value=f"R:1 P:{i+1}, Player {i+1}, Team(Team{i+1})")
        picks.append(pick)
    return picks

@pytest.fixture
def mock_successful_league(sample_draft_picks):
    """Mock a successful league connection."""
    league = Mock()
    league.draft = sample_draft_picks
    return league 