import pandas as pd
from espn_api.hockey import League
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

def fetch_player_stats(league_id, year, swid, espn_s2):
    """
    Fetches player stats for a given ESPN fantasy hockey league season.

    Args:
        league_id (int): The ESPN fantasy league ID.
        year (int): The season year.
        swid (str): The SWID cookie value.
        espn_s2 (str): The ESPN_S2 cookie value.

    Returns:
        pandas.DataFrame: DataFrame containing player names and total points,
                          or None if fetching fails.
    """
    try:
        logging.info(f"Connecting to league {league_id} for year {year}...")
        league = League(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)
        logging.info("Successfully connected to league.")
    except Exception as e:
        logging.error(f"Failed to connect to ESPN API: {e}")
        return None

    all_player_stats = []
    processed_players = set() # Keep track of players already processed
    stat_keys = set() # Keep track of all unique stat keys encountered

    logging.info("Fetching detailed player stats from team rosters...")
    for team in league.teams:
        logging.debug(f"Processing team: {team.team_name}")
        for player in team.roster:
            # Ensure we only process each unique player once
            if player.playerId not in processed_players:
                player_stats = {'Player': player.name} # Start with player name
                try:
                    # Check if stats exist and contain the target year's total stats
                    if hasattr(player, 'stats') and \
                       isinstance(player.stats, dict) and \
                       f'Total {year}' in player.stats and \
                       isinstance(player.stats[f'Total {year}'], dict) and \
                       'total' in player.stats[f'Total {year}'] and \
                       isinstance(player.stats[f'Total {year}']['total'], dict):

                        season_stats = player.stats[f'Total {year}']['total']
                        player_stats.update(season_stats) # Add all stats from the dict
                        stat_keys.update(season_stats.keys()) # Track unique stat keys
                        logging.debug(f"Successfully extracted stats for {player.name}")
                    else:
                        logging.warning(f"Could not find expected stats structure ('Total {year}' -> 'total') for player {player.name} (ID: {player.playerId}). Skipping stats for this player.")
                        # Add player with name only, other stats will be NaN later

                except Exception as e:
                    logging.error(f"Error processing stats for player {player.name} (ID: {player.playerId}): {e}")
                    # Add player with name only on error

                all_player_stats.append(player_stats)
                processed_players.add(player.playerId)

    if not all_player_stats:
        logging.warning("No player data collected. Check API connection and stats structure.")
        return pd.DataFrame(columns=['Player']) # Return empty DataFrame with Player column

    # Create DataFrame from the list of dictionaries
    # This handles players having different sets of stats (missing ones become NaN)
    stats_df = pd.DataFrame(all_player_stats)

    # Reorder columns to have 'Player' first, then sorted stat keys
    if not stats_df.empty:
        cols = ['Player'] + sorted([key for key in stat_keys if key in stats_df.columns])
        # Add any columns that might exist but weren't in stat_keys (shouldn't happen with current logic)
        other_cols = [col for col in stats_df.columns if col not in cols]
        stats_df = stats_df[cols + other_cols]

    logging.info(f"Successfully processed stats for {len(stats_df)} unique players.")
    return stats_df

if __name__ == "__main__":
    logging.info("Starting detailed player stats fetch process...")
    detailed_stats_df = fetch_player_stats(LEAGUE_ID, YEAR, SWID, ESPN_S2)

    if detailed_stats_df is not None and not detailed_stats_df.empty:
        output_file = 'player_stats.csv'
        try:
            # Fill NaN values with 0 for numeric stats before saving
            # Identify numeric columns (excluding 'Player')
            numeric_cols = detailed_stats_df.columns.difference(['Player'])
            for col in numeric_cols:
                 # Ensure column is numeric, coercing errors and filling NaNs
                 detailed_stats_df[col] = pd.to_numeric(detailed_stats_df[col], errors='coerce').fillna(0)

            detailed_stats_df.to_csv(output_file, index=False)
            logging.info(f"Detailed player stats saved successfully to {output_file}")
            print(f"\nDetailed player stats saved to {output_file}")
            print("\nFirst few rows of player stats:")
            print(detailed_stats_df.head())
        except Exception as e:
            logging.error(f"Failed to save detailed player stats to CSV: {e}")
            print(f"Error: Failed to save detailed player stats to {output_file}")
    elif detailed_stats_df is not None and detailed_stats_df.empty:
         logging.warning("Fetched stats DataFrame is empty. No data saved.")
         print("Warning: Fetched stats DataFrame is empty. No data saved.")
    else:
        logging.error("Failed to fetch detailed player stats. CSV file not created.")
        print("Error: Failed to fetch detailed player stats.")
