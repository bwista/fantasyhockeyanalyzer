from espn_api.hockey import League
import pandas as pd
import logging
import json # Added to load config
import os # Added to construct config path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION (Now loaded from user_config.json) ---

# --- SCRIPT CONFIGURATION ---
OUTPUT_FILE = 'src/data/draft_results.json' # Output file path (relative to project root where script is run)
CONFIG_FILE = '../../user_config.json' # Path relative to this script

# --- END SCRIPT CONFIGURATION

def parse_draft_results(league_id, year, swid, espn_s2):
    """
    Fetches draft results for a given ESPN fantasy hockey league season using the ESPN API.

    Args:
        league_id (int): The ESPN fantasy league ID.
        year (int): The season year.
        swid (str): The SWID cookie value.
        espn_s2 (str): The ESPN_S2 cookie value.

    Returns:
        pandas.DataFrame: DataFrame containing draft results. The main script block
            saves this data to a JSON file ('src/data/draft_results.json') with records orientation.
            DataFrame columns:
            - Round: The draft round number
            - Pick: The pick number within the round
            - Team: The team name that made the pick
            - Player: The player name that was picked
    """
    try:
        logging.info(f"Connecting to league {league_id} for year {year}...")
        league = League(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)
        logging.info("Successfully connected to league.")
    except Exception as e:
        logging.error(f"Failed to connect to ESPN API: {e}")
        return None

    try:
        logging.info("Fetching draft results...")
        draft_picks = league.draft
        logging.info(f"Successfully fetched {len(draft_picks)} draft picks.")
    except Exception as e:
        logging.error(f"Failed to fetch draft results: {e}")
        return None

    # Initialize lists to store data
    rounds = []
    pick_numbers = []
    team_names = []
    player_names = []

    # Process each pick
    for pick in draft_picks:
        # Extract round and pick number from the pick string (e.g., "R:1 P:1")
        round_pick = str(pick).split(',')[0]  # Get "R:1 P:1" part
        round_num = int(round_pick.split('R:')[1].split(' ')[0])
        pick_num = int(round_pick.split('P:')[1])

        # Extract player name (second part of the string)
        player_name = str(pick).split(',')[1].strip()

        # Extract team name (third part, remove "Team(" and ")" )
        team_name = str(pick).split(',')[2].strip()
        team_name = team_name.replace('Team(', '').replace(')', '')

        rounds.append(round_num)
        pick_numbers.append(pick_num)
        team_names.append(team_name)
        player_names.append(player_name)

    # Create DataFrame
    draft_df = pd.DataFrame({
        'Round': rounds,
        'Pick': pick_numbers,
        'Team': team_names,
        'Player': player_names
    })

    return draft_df

if __name__ == "__main__":
    # Construct the absolute path to the config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, CONFIG_FILE))

    # Load configuration from JSON file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        config = None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file: {config_path}")
        config = None
    except Exception as e:
        logging.error(f"An error occurred loading the configuration file: {e}")
        config = None

    if config:
        # Extract config values
        league_id = config.get('LEAGUE_ID')
        year = config.get('YEAR')
        swid = config.get('SWID')
        espn_s2 = config.get('ESPN_S2')

        if not all([league_id, year, swid, espn_s2]):
             logging.error("One or more required configuration keys (LEAGUE_ID, YEAR, SWID, ESPN_S2) are missing from the config file.")
        else:
            # Parse the draft results using loaded config
            draft_results = parse_draft_results(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)

            if draft_results is not None:
                # Ensure the output directory exists
                output_dir = os.path.dirname(OUTPUT_FILE)
                if output_dir and not os.path.exists(output_dir):
                    try:
                        os.makedirs(output_dir)
                        logging.info(f"Created output directory: {output_dir}")
                    except OSError as e:
                        logging.error(f"Error creating output directory {output_dir}: {e}")
                        # Exit or handle error appropriately if directory creation fails
                        draft_results = None # Prevent attempting to save

            if draft_results is not None:
                # Save to JSON
                draft_results.to_json(OUTPUT_FILE, orient='records', indent=4)
                logging.info(f"Draft results saved to {OUTPUT_FILE}")

                # Display first few rows (optional, kept for consistency)
                print("\nFirst few picks (DataFrame format):")
                print(draft_results.head())

                # Display summary (optional, kept for consistency)
                print(f"\nTotal picks: {len(draft_results)}")
                print("\nPicks by team:")
                print(draft_results['Team'].value_counts())
            else: # This corresponds to the inner 'if draft_results is not None:'
                logging.error("Failed to fetch draft results after config check. No output file created.")
    else: # This corresponds to the outer 'if config:'
        logging.error("Failed to fetch draft results. No output file created.")
