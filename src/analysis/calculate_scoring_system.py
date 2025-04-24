import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
BOX_SCORE_STATS_FILE = Path('src/data/box_score_stats.json')
OUTPUT_FILE = Path('src/data/scoring_system.json')

# Stat name mapping
STAT_MAPPING = {
    '16': 'PTS',  # Points
    '25': 'TOI',  # Time on ice
    '30': 'FOW%',  # Faceoff win percentage
    '12': 'PROD'  # Production
}

def clean_stat_name(stat: str) -> str:
    """
    Clean stat name by applying mapping if available.
    
    Args:
        stat: Original stat name
        
    Returns:
        Cleaned stat name
    """
    return STAT_MAPPING.get(stat, stat)

def load_box_score_stats(file_path: Path) -> Optional[Dict]:
    """
    Load box score stats from JSON file.
    
    Args:
        file_path: Path to the box score stats JSON file
        
    Returns:
        Dictionary containing box score stats or None if file not found/error
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Box score stats file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading box score stats: {e}")
        return None

def calculate_points_per_stat(stats: Dict, points_breakdown: Dict) -> Dict[str, float]:
    """
    Calculate points per unit for each statistical category.
    
    Args:
        stats: Dictionary of statistical values
        points_breakdown: Dictionary of points awarded
        
    Returns:
        Dictionary mapping stat categories to points per unit
    """
    points_per_stat = {}
    
    for stat, value in stats.items():
        if value != 0 and stat in points_breakdown:
            points_per_stat[stat] = points_breakdown[stat] / value
        else:
            points_per_stat[stat] = 0.0
            
    return points_per_stat

def aggregate_scoring_system(box_score_stats: List[Dict]) -> Dict[str, float]:
    """
    Aggregate scoring system across all player-game entries.
    
    Args:
        box_score_stats: List of dictionaries containing player-game stats
        
    Returns:
        Dictionary mapping stat categories to average points per unit
    """
    all_points_per_stat = []
    
    for player_game in box_score_stats:
        if 'stats' in player_game and 'points_breakdown' in player_game:
            points_per_stat = calculate_points_per_stat(
                player_game['stats'],
                player_game['points_breakdown']
            )
            all_points_per_stat.append(points_per_stat)
    
    if not all_points_per_stat:
        logging.warning("No valid player-game entries found for scoring analysis")
        return {}
    
    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame(all_points_per_stat)
    
    # Calculate mean points per stat, ignoring zeros
    scoring_system = {}
    for column in df.columns:
        non_zero_values = df[column][df[column] != 0]
        if not non_zero_values.empty:
            # Round to 1 decimal point
            value = float(non_zero_values.mean())
            if not np.isnan(value):  # Only include non-NaN values
                scoring_system[clean_stat_name(column)] = round(value, 1)
    
    return scoring_system

def save_scoring_system(scoring_system: Dict[str, float], output_file: Path) -> bool:
    """
    Save scoring system to JSON file.
    
    Args:
        scoring_system: Dictionary containing scoring system
        output_file: Path to save the scoring system
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(scoring_system, f, indent=4)
        logging.info(f"Scoring system saved to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error saving scoring system: {e}")
        return False

def main():
    """Main function to process box score stats and derive scoring system."""
    logging.info("Starting scoring system analysis...")
    
    # Load box score stats
    box_score_stats = load_box_score_stats(BOX_SCORE_STATS_FILE)
    if not box_score_stats:
        logging.error("Cannot proceed without box score stats")
        return
    
    # Calculate scoring system
    scoring_system = aggregate_scoring_system(box_score_stats)
    if not scoring_system:
        logging.error("Failed to derive scoring system")
        return
    
    # Save scoring system
    if save_scoring_system(scoring_system, OUTPUT_FILE):
        logging.info("Scoring system analysis completed successfully")
        print("\nDerived Scoring System:")
        # Sort by stat name for consistent output
        for stat in sorted(scoring_system.keys()):
            print(f"{stat}: {scoring_system[stat]:.1f} points per unit")
    else:
        logging.error("Failed to save scoring system")

if __name__ == "__main__":
    main() 