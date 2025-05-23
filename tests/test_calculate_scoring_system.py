import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
import json

# Import the functions we want to test
from src.analysis.calculate_scoring_system import (
    clean_stat_name,
    calculate_points_per_stat,
    aggregate_scoring_system,
    load_box_score_stats
)

class TestCleanStatName:
    """Test the clean_stat_name function."""
    
    def test_mapped_stat_names(self):
        """Test that mapped stat names are converted correctly."""
        assert clean_stat_name('16') == 'PTS'
        assert clean_stat_name('25') == 'TOI'
        assert clean_stat_name('30') == 'FOW%'
        assert clean_stat_name('12') == 'PROD'
    
    def test_unmapped_stat_names(self):
        """Test that unmapped stat names are returned as-is."""
        assert clean_stat_name('goals') == 'goals'
        assert clean_stat_name('assists') == 'assists'
        assert clean_stat_name('99') == '99'

class TestCalculatePointsPerStat:
    """Test the calculate_points_per_stat function."""
    
    def test_normal_calculation(self):
        """Test normal points per stat calculation."""
        stats = {'goals': 5, 'assists': 3, 'shots': 10}
        points_breakdown = {'goals': 10, 'assists': 6, 'shots': 5}
        
        result = calculate_points_per_stat(stats, points_breakdown)
        
        expected = {'goals': 2.0, 'assists': 2.0, 'shots': 0.5}
        assert result == expected
    
    def test_zero_stat_value(self):
        """Test handling of zero stat values."""
        stats = {'goals': 0, 'assists': 3}
        points_breakdown = {'goals': 10, 'assists': 6}
        
        result = calculate_points_per_stat(stats, points_breakdown)
        
        assert result['goals'] == 0.0
        assert result['assists'] == 2.0
    
    def test_stat_not_in_points_breakdown(self):
        """Test handling when stat is not in points breakdown."""
        stats = {'goals': 5, 'penalty_minutes': 2}
        points_breakdown = {'goals': 10}
        
        result = calculate_points_per_stat(stats, points_breakdown)
        
        assert result['goals'] == 2.0
        assert result['penalty_minutes'] == 0.0

class TestAggregateScoringsystem:
    """Test the aggregate_scoring_system function."""
    
    def test_normal_aggregation(self):
        """Test normal aggregation of scoring system."""
        box_score_stats = [
            {
                'stats': {'goals': 2, 'assists': 1},
                'points_breakdown': {'goals': 4, 'assists': 2}
            },
            {
                'stats': {'goals': 1, 'assists': 3},
                'points_breakdown': {'goals': 2, 'assists': 6}
            }
        ]
        
        result = aggregate_scoring_system(box_score_stats)
        
        # goals: (4/2 + 2/1) / 2 = (2 + 2) / 2 = 2.0
        # assists: (2/1 + 6/3) / 2 = (2 + 2) / 2 = 2.0
        assert result['goals'] == 2.0
        assert result['assists'] == 2.0
    
    def test_empty_input(self):
        """Test handling of empty input."""
        result = aggregate_scoring_system([])
        assert result == {}
    
    def test_missing_keys(self):
        """Test handling of records missing required keys."""
        box_score_stats = [
            {'stats': {'goals': 2}},  # Missing points_breakdown
            {'points_breakdown': {'goals': 4}},  # Missing stats
            {
                'stats': {'assists': 1},
                'points_breakdown': {'assists': 2}
            }
        ]
        
        result = aggregate_scoring_system(box_score_stats)
        
        # Only the complete record should be processed
        assert result['assists'] == 2.0
        assert 'goals' not in result

class TestLoadBoxScoreStats:
    """Test the load_box_score_stats function."""
    
    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "data"}')
    @patch("pathlib.Path.exists", return_value=True)
    def test_successful_load(self, mock_exists, mock_file):
        """Test successful loading of box score stats."""
        from pathlib import Path
        
        result = load_box_score_stats(Path("fake_path.json"))
        
        assert result == {"test": "data"}
        mock_file.assert_called_once()
    
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_file_not_found(self, mock_file):
        """Test handling of file not found."""
        from pathlib import Path
        
        result = load_box_score_stats(Path("nonexistent.json"))
        
        assert result is None
    
    @patch("builtins.open", new_callable=mock_open, read_data='invalid json')
    def test_invalid_json(self, mock_file):
        """Test handling of invalid JSON."""
        from pathlib import Path
        
        result = load_box_score_stats(Path("invalid.json"))
        
        assert result is None

# Fixtures for commonly used test data
@pytest.fixture
def sample_stats():
    """Sample stats data for testing."""
    return {'goals': 5, 'assists': 3, 'shots': 10}

@pytest.fixture
def sample_points_breakdown():
    """Sample points breakdown for testing."""
    return {'goals': 10, 'assists': 6, 'shots': 5}

@pytest.fixture
def sample_box_score_data():
    """Sample box score data for testing."""
    return [
        {
            'stats': {'16': 2, '25': 15.5},  # PTS, TOI
            'points_breakdown': {'16': 4, '25': 0}
        },
        {
            'stats': {'16': 3, '25': 18.2},
            'points_breakdown': {'16': 6, '25': 0}
        }
    ] 