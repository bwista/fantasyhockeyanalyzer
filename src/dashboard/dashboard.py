import streamlit as st
import pandas as pd
import json # Added for team mapping
import time # Added for temporary messages
import plotly.express as px # Added for plotting
import numpy as np # Added for trendline calculation
import statsmodels.api as sm # Added for OLS trendline calculation
import os # Added for file existence checks
import logging # Added for consistency if data fetching logs
import sys # Added to modify path

# --- Path Setup ---
# Ensure the project root directory (containing 'src') is in the Python path
# This is necessary when running streamlit from the root directory like: streamlit run src/dashboard/dashboard.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import data fetching functions and their constants if needed
from src.data_processing.parse_draft_results import parse_draft_results
# Import the function AND the constants needed for the call
from src.data_processing.fetch_box_score_stats import fetch_box_score_stats, START_WEEK, END_WEEK, RATE_LIMIT_DELAY
# Import the function AND the constants needed for the call
from src.data_processing.fetch_team_info import fetch_and_save_team_info, OUTPUT_DIR as TEAM_INFO_OUTPUT_DIR, OUTPUT_FILE as TEAM_INFO_OUTPUT_FILE
# Add import for new logic module
from src.dashboard.dashboard_logic import (
    ensure_data_files_exist, process_data, plot_draft_value
)

# Configure logging for dashboard (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Configuration ---
# Define paths relative to project root (which is added to sys.path)
# CONFIG_FILE_PATH = 'user_config.json'  # No longer needed - using st.secrets
DRAFT_RESULTS_FILE = 'src/data/draft_results.json' # Updated path for JSON
PLAYER_STATS_FILE = 'src/data/box_score_stats.json' # Updated path and extension
TEAM_MAPPING_FILE = 'src/data/team_mapping.json' # Added path for team Player to abbreviation mapping
POINTS_COLUMN = 'TotalPoints' # Actual points column Player after aggregation
N_PICKS_DISPLAY = 10 # Number of best/worst picks to show

# --- Helper Functions ---
def show_temporary_message(message_type, content, duration):
    """Displays a Streamlit message (info, success, etc.) for a specified duration."""
    placeholder = st.empty()
    message_func = getattr(placeholder, message_type, None)
    if message_func:
        message_func(content)
        time.sleep(duration)
        placeholder.empty()
    else:
        # Fallback or error handling if message_type is invalid
        st.error(f"Invalid message type: {message_type}")

def load_data(draft_file, stats_file, mapping_file):
    """Loads draft results (JSON), player stats (JSON), and team mapping (JSON) from files."""
    draft_df, stats_df, team_map = None, None, None # Initialize

    # --- Load Draft Data ---
    try:
        # Load from JSON instead of CSV
        draft_df = pd.read_json(draft_file, orient='records')
        # Calculate Overall Pick based on DataFrame index after loading
        draft_df['DraftPick'] = draft_df.index + 1
        show_temporary_message("success", f"Loaded draft data from {draft_file} (JSON) and added 'DraftPick'.", 0.25)
    except FileNotFoundError:
        st.error(f"Error: Draft results file not found at {draft_file}.")
        return None, None, None
    except ValueError as e: # Catch JSON parsing errors
        st.error(f"Error parsing JSON draft file {draft_file}: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading {draft_file}: {e}")
        return None, None, None

    # --- Load Stats Data ---
    try:
        stats_df = pd.read_json(stats_file)
        show_temporary_message("success", f"Loaded player stats from {stats_file}", 0.25)
    except FileNotFoundError:
        st.warning(f"Warning: Player stats file not found at {stats_file}. Proceeding without player stats.")
        # Don't return yet, try loading mapping
    except ValueError as e:
        st.error(f"Error parsing JSON stats file {stats_file}: {e}")
        # Don't return yet, try loading mapping
    except Exception as e:
        st.error(f"Error loading {stats_file}: {e}")
        # Don't return yet, try loading mapping

    # --- Load Team Mapping ---
    try:
        with open(mapping_file, 'r') as f:
            team_map = json.load(f)
        show_temporary_message("success", f"Loaded team mapping from {mapping_file}", 0.25)
    except FileNotFoundError:
        st.warning(f"Warning: Team mapping file not found at {mapping_file}. Cannot map draft team names to abbreviations.")
        # Proceed without mapping if it's missing
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON mapping file {mapping_file}: {e}")
        # Proceed without mapping if it's corrupted
    except Exception as e:
        st.error(f"Error loading {mapping_file}: {e}")
        # Proceed without mapping

    return draft_df, stats_df, team_map

def calculate_value(df, points_col):
    """
    Calculates PointsRank and ValueScore on the processed DataFrame.
    ValueScore is only calculated for 'drafted' records.
    """
    if df is None or df.empty:
        st.warning("Cannot calculate value: Input DataFrame is empty.")
        return None

    # Ensure required columns exist
    required_cols = [points_col, 'DraftPick', 'acquisition_type']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Cannot calculate value: Missing one or more required columns: {required_cols}")
        # Add missing columns with default values if possible, or return None
        if points_col not in df.columns: df[points_col] = 0
        if 'DraftPick' not in df.columns: df['DraftPick'] = np.nan
        if 'acquisition_type' not in df.columns: df['acquisition_type'] = 'unknown'
        # Consider returning None if critical columns are missing and cannot be defaulted reasonably
        # return None

    # Ensure points column is numeric and fill NaNs
    df[points_col] = pd.to_numeric(df[points_col], errors='coerce').fillna(0)
    # Ensure Overall Pick is numeric (can be NaN for non-drafted)
    df['DraftPick'] = pd.to_numeric(df['DraftPick'], errors='coerce')

    # Calculate Points Rank based on overall TotalPoints (higher points = better rank)
    df['PointsRank'] = df[points_col].rank(method='dense', ascending=False)

    # Draft Rank is the overall pick number (will be NaN for non-drafted)
    df['DraftRank'] = df['DraftPick']

    # Calculate Value Score only for drafted players
    # Initialize ValueScore column with NaN
    df['ValueScore'] = np.nan

    # Apply calculation where acquisition_type is 'drafted' and DraftRank is valid
    drafted_mask = (df['acquisition_type'] == 'drafted') & df['DraftRank'].notna() & df['PointsRank'].notna()
    df.loc[drafted_mask, 'ValueScore'] = df.loc[drafted_mask, 'DraftRank'] - df.loc[drafted_mask, 'PointsRank']

    # No sorting is done here - sorting happens in display sections
    return df

# --- Helper Function for Acquisition Type ---
def determine_acquisition_type(group):
    """
    Determines the acquisition type ('drafted' or 'waiver') for each player-team record within a player's group.
    Assumes the group DataFrame is sorted by 'FirstWeek'.
    """
    group = group.sort_values('FirstWeek') # Ensure sorting within the group
    acquisition_types = []
    is_drafted = False # Flag to track if the player's initial 'drafted' record has been assigned

    for i, row in group.iterrows():
        current_type = 'waiver' # Default to waiver

        # Check conditions for the *first* record of the player
        if i == group.index[0]:
            if pd.notna(row['DraftPick']) and row['team_abbrev'] == row['DraftingTeamAbbrev']: # Logic remains based on Abbrev
                current_type = 'drafted'
                is_drafted = True
            # Else: remains 'waiver' (undrafted or drafted but first record doesn't match drafting team)

        # For subsequent records, they are always 'waiver'
        # (This logic is implicitly handled by defaulting to 'waiver' and only changing the first record if conditions met)

        acquisition_types.append(current_type)

    group['acquisition_type'] = acquisition_types
    return group

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title('🏒 Fantasy Hockey Season Analysis')

# --- Create Tabs ---
draft_tab, team_tab = st.tabs(["📊 Draft", "🏒 Team"])

with draft_tab:
    # --- Load User Configuration from Streamlit Secrets ---
    config_placeholder = st.empty()  # Create a placeholder for the config section
    with config_placeholder.container():
        st.subheader("Configuration")
        try:
            # Extract essential config values from st.secrets
            league_id = st.secrets["LEAGUE_ID"]
            year = st.secrets["YEAR"]
            swid = st.secrets["SWID"]
            espn_s2 = st.secrets["ESPN_S2"]
            
            # Create config dict for compatibility with existing code
            config = {
                'LEAGUE_ID': league_id,
                'YEAR': year,
                'SWID': swid,
                'ESPN_S2': espn_s2
            }
            
            st.success("Loaded configuration from Streamlit secrets")
        except KeyError as e:
            st.error(f"ERROR: Missing required secret: {e}. Please check your secrets configuration.")
            st.stop()
        except Exception as e:
            st.error(f"ERROR: An unexpected error occurred loading secrets: {e}")
            st.stop()

    # If we reach here, config is loaded and valid, so clear the section:
    config_placeholder.empty()

    if not all([league_id, year, swid, espn_s2]):
        st.error("ERROR: Configuration file is missing one or more required keys: LEAGUE_ID, YEAR, SWID, ESPN_S2.")
        st.stop()

    # --- Data File Checks and Generation ---
    data_loading_placeholder = st.empty()  # Create a placeholder for the data loading section
    with data_loading_placeholder.container():
        st.subheader("Data Loading & Preparation")

        # Use new modular function for all data file checks and loading
        draft_df, stats_df, team_map = ensure_data_files_exist(
            config,
            DRAFT_RESULTS_FILE,
            PLAYER_STATS_FILE,
            TEAM_MAPPING_FILE,
            parse_draft_results,
            fetch_box_score_stats,
            fetch_and_save_team_info,
            START_WEEK,
            END_WEEK,
            RATE_LIMIT_DELAY,
            TEAM_INFO_OUTPUT_DIR,
            TEAM_INFO_OUTPUT_FILE,
            st=st
        )
    # If we reach here, all data files are loaded/generated, so clear the section:
    data_loading_placeholder.empty()

    # Proceed only if draft and stats data are loaded
    if draft_df is not None and stats_df is not None:
        # --- Data Processing Steps ---
        final_df, value_df = process_data(draft_df, stats_df, team_map, st=st)

        # --- Display Results ---
        if value_df is not None:
            plot_draft_value(value_df, st) # Use modular plotting function

            # --- Overall Value Tables (Based on Drafted Players) ---
            st.subheader("Overall Draft Value Analysis")
            st.markdown("_Based on players' performance relative to their draft position (`ValueScore = DraftRank - PointsRank`). Higher is better._")

            # Filter for drafted records for value tables
            drafted_value_df = value_df[value_df['acquisition_type'] == 'drafted'].sort_values(by='ValueScore', ascending=False)

            col1, col2 = st.columns(2)
            # Define columns for concise display tables
            display_cols_value = ['Pick', 'Player', 'Pos', 'Team', 'TeamPoints', 'PointsRank', 'ValueScore']

            with col1:
                st.markdown(f"**Top {N_PICKS_DISPLAY} Best Value Picks**")
                if not drafted_value_df.empty:
                     st.dataframe(drafted_value_df.head(N_PICKS_DISPLAY)[display_cols_value], use_container_width=True, hide_index=True)
                else:
                     st.info("No 'drafted' players found.")

            with col2:
                st.markdown(f"**Top {N_PICKS_DISPLAY} Worst Value Picks**")
                if not drafted_value_df.empty:
                     st.dataframe(drafted_value_df.tail(N_PICKS_DISPLAY).sort_values(by='ValueScore', ascending=True)[display_cols_value], use_container_width=True, hide_index=True)
                else:
                     st.info("No 'drafted' players found matching the criteria (including hold duration).")

            # --- Team-Specific Draft Value Analysis (Based on Drafted Players) ---
            st.subheader("Team-Specific Draft Value Analysis")
            # Get drafting teams from the drafted subset
            drafting_teams = sorted(drafted_value_df['Team'].dropna().unique()) # Use Team
            if drafting_teams:
                 selected_draft_team = st.selectbox('Select Drafting Team:', drafting_teams) # Shows full names
                 if selected_draft_team:
                     team_value_df = drafted_value_df[drafted_value_df['Team'] == selected_draft_team].copy() # Filter by Team
                     st.markdown(f"**Draft Value Analysis for {selected_draft_team}**")

                     # Show only half as many picks for team-specific analysis
                     TEAM_N_PICKS_DISPLAY = max(1, N_PICKS_DISPLAY // 2)

                     col3, col4 = st.columns(2)
                     with col3:
                         st.markdown(f"*Top {TEAM_N_PICKS_DISPLAY} Best Value Picks*")
                         st.dataframe(team_value_df.head(TEAM_N_PICKS_DISPLAY)[display_cols_value], use_container_width=True, hide_index=True)
                     with col4:
                         st.markdown(f"*Top {TEAM_N_PICKS_DISPLAY} Worst Value Picks*")
                         st.dataframe(team_value_df.tail(TEAM_N_PICKS_DISPLAY).sort_values(by='ValueScore', ascending=True)[display_cols_value], use_container_width=True, hide_index=True)
            else:
                st.info("No drafting teams found in the data for drafted players.")

            # --- NEW: Best Waiver/Trade Acquisitions Section ---
            st.header("Waiver Wire & Trade Analysis")

            # Filter for waiver records
            waiver_records_df = value_df[value_df['acquisition_type'] == 'waiver'].copy()

            # 1. Top Overall Acquisitions (Unique Players by TotalPoints)
            st.subheader(f"Top {N_PICKS_DISPLAY} Overall Acquisitions (by Season Total Points)")
            st.markdown("_Players acquired via waiver/trade, ranked by their total points scored across the entire season._")
            if not waiver_records_df.empty:
                # Get unique players acquired via waiver, include their original Team
                unique_waiver_players = waiver_records_df[['Player','Pos', 'TeamPoints', 'PickupTeamName','FirstWeek','LastWeek']].drop_duplicates(subset=['Player'])

                unique_waiver_players['Duration'] = unique_waiver_players['LastWeek'] - unique_waiver_players['FirstWeek'] #calculate duration
                unique_waiver_players['AvgPointsPerWeek'] = unique_waiver_players['TeamPoints'] / unique_waiver_players['Duration'] #calculate average points per week

                # Sort them by TeamPoints
                top_overall_acquisitions = unique_waiver_players.sort_values(by='TeamPoints', ascending=False)
                # Display
                display_cols_overall_acq = ['Player','Pos', 'PickupTeamName', 'TeamPoints', 'AvgPointsPerWeek', 'Duration']
                st.dataframe(top_overall_acquisitions.head(N_PICKS_DISPLAY)[display_cols_overall_acq], hide_index=True, use_container_width=True)
            else:
                st.info("No waiver/trade acquisitions found.")

            # 2. Top Team Acquisitions (by TeamPoints scored for that team)
            st.subheader("Top Acquisitions by Acquiring Team")
            st.markdown("_Players acquired via waiver/trade by each team, ranked by points scored *for that specific team* after acquisition._")
            if not waiver_records_df.empty:
                # Get acquiring teams
                acquiring_teams = sorted(waiver_records_df['team_abbrev'].dropna().unique())
                if acquiring_teams:
                    selected_acq_team = st.selectbox('Select Acquiring Team:', acquiring_teams)
                    if selected_acq_team:
                        team_acquisitions_df = waiver_records_df[waiver_records_df['team_abbrev'] == selected_acq_team].copy()

                        team_acquisitions_df['Duration'] = team_acquisitions_df['LastWeek'] - team_acquisitions_df['FirstWeek'] #calculate duration
                        team_acquisitions_df['AvgPointsPerWeek'] = team_acquisitions_df['TeamPoints'] / team_acquisitions_df['Duration'] #calculate average points per week
                        # Sort by points scored for *this* team
                        team_acquisitions_df = team_acquisitions_df.sort_values(by='TeamPoints', ascending=False)
                        st.markdown(f"**Top {N_PICKS_DISPLAY} Acquisitions for {selected_acq_team} (by Points for Team)**")
                        display_cols_team_acq = ['Player','Pos', 'TeamPoints', 'AvgPointsPerWeek', 'Duration'] # Show points for team and duration
                        st.dataframe(team_acquisitions_df.head(N_PICKS_DISPLAY)[display_cols_team_acq], hide_index=True, use_container_width=True)
                else:
                    st.info("No teams found who made waiver/trade acquisitions.")
            else:
                st.info("No waiver/trade acquisitions found.")

            # --- Updated Full Data Table ---
            st.subheader("Full Processed Data")
            st.markdown("_Includes all player-team records with acquisition type._")
            with st.expander("Show Full Data Table"):
                # Define columns for the full table display
                all_cols_ordered = [
                    'Player', 'team_abbrev', 'acquisition_type', 'TeamPoints', 'TotalPoints', # Core Info
                    'FirstWeek', 'LastWeek', # Stint Info
                    'Pick', 'Team', 'DraftingTeamAbbrev', # Draft Info (if applicable)
                    'PointsRank', 'ValueScore' # Ranks & Value (if applicable)
                ]
                # Add any remaining columns automatically
                remaining_cols = [col for col in value_df.columns if col not in all_cols_ordered]
                # Display using the final value_df
                st.dataframe(value_df[all_cols_ordered + remaining_cols], use_container_width=True, hide_index=True)

    elif draft_df is None:
         st.warning("Draft data could not be loaded. Cannot display dashboard.")
    else: # stats_df is None
         st.warning("Stats data could not be loaded. Cannot perform analysis.") # Added specific message for missing stats

with team_tab:
    st.header("Team Analysis")
    st.info("Team analysis features coming soon! This tab will contain team-specific insights and performance metrics.")
    
    # Placeholder content for the Team tab
    st.markdown("""
    ### Planned Features:
    - Team performance comparison
    - Roster composition analysis
    - Head-to-head matchup insights
    - Season progression tracking
    - Lineup optimization suggestions
    """)
