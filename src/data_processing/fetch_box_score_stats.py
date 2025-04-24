import json
from espn_api.hockey import League
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- USER CONFIGURATION (Copied from fetch_player_stats.py) ---
# Replace with your actual league details and credentials
LEAGUE_ID = 52632018
YEAR = 2025
SWID = '{EC3394CD-9286-4EBB-BFF0-BE0BDFBCB79F}' # Replace with your SWID Cookie
ESPN_S2 = 'AEBS5WA%2Bo11dLS2wzax2UxfLN3h9JTqNsqBtLbR%2BIEQXSBfIKZvcoiwCmKH2DjQb2jcwP3bYydJYQH9up9sJERKnvikLlsCGHnTtEtkVf49epcUZLaOSUmhCgoZwthxSORcY2TFbVvcf4hu3K9rHmk454ADEUr%2BUarxBYh725lOGgjlzQZ97qYHM139NkD%2FnzSU4QmwWBYLiVLIX8ImweEzFP2Knx5z0auEzL3F6jcsdmWKBzX7mFqt6hs4PtYpDUHhnGX88UKI3I1QSazxxkKMLXnhZ5HulnlEd7ZZTYF8r%2BQ%3D%3D' # Replace with your ESPN_S2 Cookie
# --- END USER CONFIGURATION ---

# --- SCRIPT CONFIGURATION ---
OUTPUT_FILE = 'box_score_stats.json' # Output file path (relative to project root where script is run)
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
                for player in box.home_lineup + box.away_lineup:
                    player_data = {
                        'week': week,
                        'team_id': player.owner_team.team_id if hasattr(player, 'owner_team') else None, # Get team ID if available
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
                            logging.debug(f"Extracted stats for {player.name} in week {week}")
                        else:
                             logging.warning(f"Could not find expected 'total' stats structure for player {player.name} (ID: {player.playerId}) in week {week}. Stats dict will be empty.")

                    except Exception as e:
                        logging.error(f"Error extracting raw stats for player {player.name} (ID: {player.playerId}) in week {week}: {e}")

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

if __name__ == "__main__":
    logging.info("Starting box score stats fetch process...")
    box_stats = fetch_box_score_stats(LEAGUE_ID, YEAR, SWID, ESPN_S2, START_WEEK, END_WEEK, RATE_LIMIT_DELAY)

    if box_stats:
        try:
            logging.info(f"Attempting to save data to project root: {OUTPUT_FILE}...")
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(box_stats, f, indent=4)
            logging.info(f"Box score stats saved successfully to project root: {OUTPUT_FILE}")
            print(f"\nBox score stats saved to {OUTPUT_FILE}")
            print(f"Total player-game entries saved: {len(box_stats)}")
        except Exception as e:
            logging.error(f"Failed to save box score stats to JSON: {e}")
            print(f"Error: Failed to save box score stats to {OUTPUT_FILE}")
    else:
        logging.warning("No box score data collected. JSON file not created.")
        print("Warning: No box score data collected. JSON file not created.")
