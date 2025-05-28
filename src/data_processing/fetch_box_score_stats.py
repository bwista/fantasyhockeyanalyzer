import json
from espn_api.hockey import League
import logging
import time
import os # Added for config path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION (Now loaded from user_config.json) ---

# --- SCRIPT CONFIGURATION ---
OUTPUT_FILE = 'src/data/box_score_stats.json' # Output file path (relative to project root where script is run)
CONFIG_FILE = '../../user_config.json' # Path relative to this script
START_WEEK = 1 # Week to start fetching data from
END_WEEK = None # Set to None to fetch up to the current week, or specify an end week number
RATE_LIMIT_DELAY = 2 # Seconds to wait between fetching weeks to avoid rate limiting
# --- END SCRIPT CONFIGURATION ---

def fetch_box_score_stats(league_id, year, swid, espn_s2, start_week, end_week, delay):
    """
    Fetches detailed player stats from box scores for a given ESPN fantasy hockey league season.

    Args:
        league_id (int): The ESPN fantasy league ID.
        year (int): The season year.
        swid (str): The SWID cookie value.
        espn_s2 (str): The ESPN_S2 cookie value.
        start_week (int): The first matchup period (week) to fetch.
        end_week (int or None): The last matchup period (week) to fetch. If None, fetches up to current week.
        delay (int): Seconds to wait between weekly requests.

    Returns:
        list: A list of dictionaries, where each dictionary contains stats for one player in one game.
              Returns an empty list if fetching fails or no data is found.
    """
    try:
        logging.info(f"Connecting to league {league_id} for year {year}...")
        league = League(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)
        logging.info("Successfully connected to league.")
    except Exception as e:
        logging.error(f"Failed to connect to ESPN API: {e}")
        return []

    if end_week is None:
        try:
            # Determine the last matchup period from the settings
            matchup_period_ids = [int(k) for k in league.settings.matchup_periods.keys()]
            if not matchup_period_ids:
                 raise ValueError("Matchup periods dictionary is empty.")
            end_week = max(matchup_period_ids)
            logging.info(f"End week not specified, fetching up to the last matchup period found in settings: {end_week}")
        except Exception as e:
            logging.error(f"Could not determine the last matchup period from league settings: {e}")
            return []

    all_box_score_stats = []

    logging.info(f"Fetching box scores from week {start_week} to {end_week}...")
    for week in range(start_week, end_week + 1):
        logging.info(f"Fetching data for week {week}...")
        try:
            box_scores = league.box_scores(matchup_period=week)
            logging.info(f"Successfully fetched {len(box_scores)} box scores for week {week}.")

            for box in box_scores:
                logging.debug(f"Processing box score: {box.home_team.team_name} vs {box.away_team.team_name}")
                # Extract team abbreviations directly from the box score object
                home_team_abbrev = box.home_team.team_abbrev
                away_team_abbrev = box.away_team.team_abbrev

                # Process home team lineup
                for player in box.home_lineup:
                    player_data = {
                        'week': week,
                        'team_abbrev': home_team_abbrev, # Assign home team abbreviation
                        'player_id': player.playerId,
                        'name': player.name,
                        'position': player.position,
                        'slot_position': player.slot_position,
                        'pro_opponent': player.pro_opponent,
                        'game_played': player.game_played,
                        'total_points': player.points,
                        'points_breakdown': player.points_breakdown,
                        'stats': {} # Initialize stats dict
                    }

                    # Extract raw stats - structure might vary, attempt common path
                    try:
                        # The example showed stats under '05null', let's try that first
                        # It might also be directly under 'total' for some stat views
                        raw_stats = None
                        if hasattr(player, 'stats') and isinstance(player.stats, dict):
                            if '05null' in player.stats and isinstance(player.stats['05null'], dict) and 'total' in player.stats['05null']:
                                raw_stats = player.stats['05null']['total']
                            elif 'total' in player.stats: # Fallback if '05null' isn't present
                                raw_stats = player.stats['total']
                            # Add more potential paths if needed based on API variations

                        if raw_stats and isinstance(raw_stats, dict):
                            player_data['stats'] = raw_stats
                            logging.debug(f"Extracted stats for {player.name} (Team: {home_team_abbrev}) in week {week}")
                        else:
                             logging.warning(f"Could not find expected 'total' stats structure for player {player.name} (ID: {player.playerId}, Team: {home_team_abbrev}) in week {week}. Stats dict will be empty.")

                    except Exception as e:
                        logging.error(f"Error extracting raw stats for player {player.name} (ID: {player.playerId}, Team: {home_team_abbrev}) in week {week}: {e}")

                    all_box_score_stats.append(player_data)

                # Process away team lineup
                for player in box.away_lineup:
                    player_data = {
                        'week': week,
                        'team_abbrev': away_team_abbrev, # Assign away team abbreviation
                        'player_id': player.playerId,
                        'name': player.name,
                        'position': player.position,
                        'slot_position': player.slot_position,
                        'pro_opponent': player.pro_opponent,
                        'game_played': player.game_played,
                        'total_points': player.points,
                        'points_breakdown': player.points_breakdown,
                        'stats': {} # Initialize stats dict
                    }

                    # Extract raw stats - structure might vary, attempt common path
                    try:
                        # The example showed stats under '05null', let's try that first
                        # It might also be directly under 'total' for some stat views
                        raw_stats = None
                        if hasattr(player, 'stats') and isinstance(player.stats, dict):
                            if '05null' in player.stats and isinstance(player.stats['05null'], dict) and 'total' in player.stats['05null']:
                                raw_stats = player.stats['05null']['total']
                            elif 'total' in player.stats: # Fallback if '05null' isn't present
                                raw_stats = player.stats['total']
                            # Add more potential paths if needed based on API variations

                        if raw_stats and isinstance(raw_stats, dict):
                            player_data['stats'] = raw_stats
                            logging.debug(f"Extracted stats for {player.name} (Team: {away_team_abbrev}) in week {week}")
                        else:
                             logging.warning(f"Could not find expected 'total' stats structure for player {player.name} (ID: {player.playerId}, Team: {away_team_abbrev}) in week {week}. Stats dict will be empty.")

                    except Exception as e:
                        logging.error(f"Error extracting raw stats for player {player.name} (ID: {player.playerId}, Team: {away_team_abbrev}) in week {week}: {e}")

                    all_box_score_stats.append(player_data)

            # Wait before fetching the next week to avoid rate limiting
            if week < end_week:
                logging.info(f"Waiting {delay} seconds before fetching next week...")
                time.sleep(delay)

        except Exception as e:
            logging.error(f"Failed to fetch or process box scores for week {week}: {e}")
            # Optionally continue to the next week or break the loop
            # continue

    logging.info(f"Finished fetching data. Collected {len(all_box_score_stats)} player-game entries.")
    return all_box_score_stats

def load_config():
    """Load configuration from either Streamlit secrets or JSON file."""
    # Try to load from Streamlit secrets first (if running in Streamlit)
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            config = {
                'LEAGUE_ID': st.secrets["LEAGUE_ID"],
                'YEAR': st.secrets["YEAR"],
                'SWID': st.secrets["SWID"],
                'ESPN_S2': st.secrets["ESPN_S2"]
            }
            logging.info("Loaded configuration from Streamlit secrets")
            return config
    except (ImportError, KeyError, AttributeError):
        # Fall back to JSON file if Streamlit is not available or secrets are missing
        pass
    
    # Load configuration from JSON file (fallback)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, CONFIG_FILE))
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file: {config_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred loading the configuration file: {e}")
        return None

if __name__ == "__main__":
    # Load configuration
    config = load_config()

    if config:
        # Extract config values
        league_id = config.get('LEAGUE_ID')
        year = config.get('YEAR')
        swid = config.get('SWID')
        espn_s2 = config.get('ESPN_S2')

        if not all([league_id, year, swid, espn_s2]):
             logging.error("One or more required configuration keys (LEAGUE_ID, YEAR, SWID, ESPN_S2) are missing from the config file.")
        else:
            logging.info("Starting box score stats fetch process...")
            # Use loaded config and script defaults for week/delay
            box_stats = fetch_box_score_stats(
                league_id=league_id,
                year=year,
                swid=swid,
                espn_s2=espn_s2,
                start_week=START_WEEK,
                end_week=END_WEEK,
                delay=RATE_LIMIT_DELAY
            )

            if box_stats:
                 # Ensure the output directory exists
                 output_dir = os.path.dirname(OUTPUT_FILE)
                 if output_dir and not os.path.exists(output_dir):
                     try:
                         os.makedirs(output_dir)
                         logging.info(f"Created output directory: {output_dir}")
                     except OSError as e:
                         logging.error(f"Error creating output directory {output_dir}: {e}")
                         box_stats = None # Prevent attempting to save

            if box_stats: # Re-check in case directory creation failed
                try:
                    logging.info(f"Attempting to save data to: {OUTPUT_FILE}...")
                    with open(OUTPUT_FILE, 'w') as f:
                        json.dump(box_stats, f, indent=4)
                    # Correctly indented block after successful save
                    logging.info(f"Box score stats saved successfully to: {OUTPUT_FILE}")
                    print(f"\nBox score stats saved to {OUTPUT_FILE}")
                    print(f"Total player-game entries saved: {len(box_stats)}")
                except Exception as e:
                    logging.error(f"Failed to save box score stats to JSON: {e}")
                    print(f"Error: Failed to save box score stats to {OUTPUT_FILE}")
            else: # This corresponds to the 'if box_stats:' check (line ~216)
                 logging.warning("No box score data collected or directory creation failed. JSON file not created.")
                 print("Warning: No box score data collected or directory creation failed. JSON file not created.")
    else: # This corresponds to the outer 'if config:'
        logging.error("Failed to load configuration. Cannot fetch box score stats.")
