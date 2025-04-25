import json
import os
import logging
from espn_api.hockey import League

# Configure logging (same style as fetch_box_score_stats.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- USER CONFIGURATION (Copied from fetch_box_score_stats.py) ---
# Replace with your actual league details and credentials
LEAGUE_ID = 52632018
YEAR = 2025
SWID = '{EC3394CD-9286-4EBB-BFF0-BE0BDFBCB79F}' # Replace with your SWID Cookie
ESPN_S2 = 'AEBS5WA%2Bo11dLS2wzax2UxfLN3h9JTqNsqBtLbR%2BIEQXSBfIKZvcoiwCmKH2DjQb2jcwP3bYydJYQH9up9sJERKnvikLlsCGHnTtEtkVf49epcUZLaOSUmhCgoZwthxSORcY2TFbVvcf4hu3K9rHmk454ADEUr%2BUarxBYh725lOGgjlzQZ97qYHM139NkD%2FnzSU4QmwWBYLiVLIX8ImweEzFP2Knx5z0auEzL3F6jcsdmWKBzX7mFqt6hs4PtYpDUHhnGX88UKI3I1QSazxxkKMLXnhZ5HulnlEd7ZZTYF8r%2BQ%3D%3D' # Replace with your ESPN_S2 Cookie
# --- END USER CONFIGURATION ---

# --- SCRIPT CONFIGURATION ---
OUTPUT_DIR = 'src/data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'team_mapping.json') # Output file for the name->abbreviation map
# --- END SCRIPT CONFIGURATION ---

def fetch_and_save_team_info(league_id, year, swid, espn_s2, output_dir, output_file):
    """
    Connects to the ESPN league, extracts team information (ID, name, abbreviation),
    and saves a mapping of team names to abbreviations to a JSON file.
    """
    logging.info(f"Attempting to connect to ESPN Hockey League ID: {league_id} for year: {year}...")

    try:
        # Initialize the League object using provided credentials
        league = League(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)
        logging.info("Successfully connected to the league.")

    except Exception as e:
        logging.error(f"Error connecting to the ESPN league: {e}")
        logging.error("Please ensure the LEAGUE_ID, YEAR, SWID, and ESPN_S2 are correct.")
        return False # Indicate failure

    team_mapping = {}
    logging.info("Extracting team information (ID, Name, Abbreviation)...")

    if not league.teams:
        logging.warning("No teams found in the league object. Cannot generate mapping.")
        return False # Indicate failure

    all_teams_processed = True
    for team in league.teams:
        try:
            team_id = team.team_id
            team_name = team.team_name
            team_abbrev = team.team_abbrev

            if team_name and team_abbrev:
                team_mapping[team_name] = team_abbrev
                logging.info(f"  - ID: {team_id}, Name: '{team_name}', Abbrev: '{team_abbrev}' -> Added to map.")
            else:
                logging.warning(f"  - Warning: Missing name or abbreviation for team ID {team_id}. Skipping.")
                all_teams_processed = False # Flag that some data might be missing
        except AttributeError as e:
            logging.warning(f"  - Warning: Could not access expected attribute for a team object: {e}. Skipping.")
            all_teams_processed = False
        except Exception as e:
             logging.error(f"  - Error processing team: {e}")
             all_teams_processed = False

    if not team_mapping:
        logging.error("No valid team mappings were extracted. Aborting file write.")
        return False # Indicate failure

    logging.info(f"\nGenerated {len(team_mapping)} team name -> abbreviation mappings.")
    if not all_teams_processed:
        logging.warning("Note: Some teams may have been skipped due to missing data or errors.")

    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logging.error(f"Error creating output directory {output_dir}: {e}")
        return False # Cannot proceed without the directory

    # Write the mapping to the JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(team_mapping, f, indent=2) # Use indent=2 for readability
        logging.info(f"Successfully wrote team mapping to: {output_file}")
        print(f"\nTeam name to abbreviation mapping saved to: {output_file}")
        return True # Indicate success
    except IOError as e:
        logging.error(f"Error writing mapping file to {output_file}: {e}")
        print(f"Error: Failed to write mapping file to {output_file}")
        return False # Indicate failure
    except Exception as e:
        logging.error(f"An unexpected error occurred during file writing: {e}")
        print(f"Error: An unexpected error occurred writing to {output_file}")
        return False # Indicate failure

if __name__ == "__main__":
    logging.info("Starting team info fetch process...")
    success = fetch_and_save_team_info(LEAGUE_ID, YEAR, SWID, ESPN_S2, OUTPUT_DIR, OUTPUT_FILE)

    if success:
        logging.info("Team info fetch process completed successfully.")
    else:
        logging.error("Team info fetch process failed.")