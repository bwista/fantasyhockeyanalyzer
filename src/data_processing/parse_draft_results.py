from espn_api.hockey import League
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- USER CONFIGURATION ---
# Replace with your actual league details and credentials
LEAGUE_ID = 52632018
YEAR = 2025
SWID = '{EC3394CD-9286-4EBB-BFF0-BE0BDFBCB79F}' # Replace with your SWID Cookie
ESPN_S2 = 'AEBS5WA%2Bo11dLS2wzax2UxfLN3h9JTqNsqBtLbR%2BIEQXSBfIKZvcoiwCmKH2DjQb2jcwP3bYydJYQH9up9sJERKnvikLlsCGHnTtEtkVf49epcUZLaOSUmhCgoZwthxSORcY2TFbVvcf4hu3K9rHmk454ADEUr%2BUarxBYh725lOGgjlzQZ97qYHM139NkD%2FnzSU4QmwWBYLiVLIX8ImweEzFP2Knx5z0auEzL3F6jcsdmWKBzX7mFqt6hs4PtYpDUHhnGX88UKI3I1QSazxxkKMLXnhZ5HulnlEd7ZZTYF8r%2BQ%3D%3D' # Replace with your ESPN_S2 Cookie
# --- END USER CONFIGURATION ---

def parse_draft_results(league_id=LEAGUE_ID, year=YEAR, swid=SWID, espn_s2=ESPN_S2):
    """
    Fetches draft results for a given ESPN fantasy hockey league season using the ESPN API.

    Args:
        league_id (int): The ESPN fantasy league ID.
        year (int): The season year.
        swid (str): The SWID cookie value.
        espn_s2 (str): The ESPN_S2 cookie value.

    Returns:
        pandas.DataFrame: DataFrame containing draft results with columns:
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
    # Parse the draft results
    draft_results = parse_draft_results()
    
    if draft_results is not None:
        # Save to CSV
        draft_results.to_csv('../../draft_results.csv', index=False)
        
        # Display first few rows
        print("\nFirst few picks:")
        print(draft_results.head())
        
        # Display summary
        print(f"\nTotal picks: {len(draft_results)}")
        print("\nPicks by team:")
        print(draft_results['Team'].value_counts())
    else:
        print("Error: Failed to fetch draft results.")
