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
from datetime import datetime, timezone

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
from src.data_processing.fetch_team_schedule import fetch_and_save_team_schedule, OUTPUT_FILE as TEAM_SCHEDULE_FILE
# Add import for new logic module
from src.dashboard.dashboard_logic import (
    ensure_data_files_exist, process_data, plot_draft_value, team_schedule_to_dataframe,
    plot_matchup_scores_by_period, compute_duration_and_avg, get_acquiring_teams
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

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.header('🏒 Fantasy Hockey Season Analysis')

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
        draft_df, stats_df, team_map, schedule_payload = ensure_data_files_exist(
            config,
            DRAFT_RESULTS_FILE,
            PLAYER_STATS_FILE,
            TEAM_MAPPING_FILE,
            TEAM_SCHEDULE_FILE,
            parse_draft_results,
            fetch_box_score_stats,
            fetch_and_save_team_info,
            fetch_and_save_team_schedule,
            START_WEEK,
            END_WEEK,
            RATE_LIMIT_DELAY,
            TEAM_INFO_OUTPUT_DIR,
            TEAM_INFO_OUTPUT_FILE,
            st=st
        )
    # If we reach here, all data files are loaded/generated, so clear the section:
    data_loading_placeholder.empty()

    schedule_df = team_schedule_to_dataframe(schedule_payload)
    schedule_generated_at = None
    if schedule_df is not None and not schedule_df.empty:
        schedule_generated_at = schedule_df['generatedAt'].dropna().max()

    # Proceed only if draft and stats data are loaded
    if draft_df is not None and stats_df is not None:
        # --- Data Processing Steps ---
        try:
            final_df, value_df = process_data(draft_df, stats_df, team_map, st=st)
        except Exception as e:
            st.error(f"Data processing failed: {e}")
            st.stop()

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

                unique_waiver_players = compute_duration_and_avg(unique_waiver_players)

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
                acquiring_teams = get_acquiring_teams(waiver_records_df)
                if acquiring_teams:
                    selected_acq_team = st.selectbox('Select Acquiring Team:', acquiring_teams)
                    if selected_acq_team:
                        team_acquisitions_df = waiver_records_df[waiver_records_df['PickupTeamName'] == selected_acq_team].copy()

                        team_acquisitions_df = compute_duration_and_avg(team_acquisitions_df)
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
    if schedule_df is None or schedule_df.empty:
        st.info("Team schedule data is not available. Run data preparation to fetch the latest schedule.")
    else:
        content_col, filter_col = st.columns([3, 1])

        with filter_col:
            st.subheader("Filters")
            team_names = sorted(schedule_df['teamName'].dropna().unique())
            if not team_names:
                st.info("No teams found in schedule data.")
                st.stop()

            selected_team_name = st.selectbox("Select Team", team_names, index=0)
            include_playoffs = st.checkbox(
                "Include playoff matchups",
                value=True,
                help="Turn off to focus on regular season matchups."
            )

            selected_team_df = schedule_df[schedule_df['teamName'] == selected_team_name].copy()
            if not include_playoffs:
                selected_team_df = selected_team_df[~selected_team_df['isPlayoff']]

            available_periods = sorted([int(mp) for mp in selected_team_df['matchupPeriod'].dropna().unique().tolist()])
            if available_periods:
                min_period, max_period = min(available_periods), max(available_periods)
                selected_range = st.slider(
                    "Filter by Matchup Period:",
                    min_value=min_period,
                    max_value=max_period,
                    value=(min_period, max_period),
                    key=f"matchup_period_slider_{selected_team_name}"
                )
                start_period, end_period = selected_range
                filtered_df = selected_team_df[
                    (selected_team_df['matchupPeriod'] >= start_period) &
                    (selected_team_df['matchupPeriod'] <= end_period)
                ].copy()
            else:
                filtered_df = selected_team_df.copy()

        with content_col:
            st.subheader(f"Team Report: {selected_team_name}")
            if schedule_generated_at is not None and not pd.isna(schedule_generated_at):
                generated_ts = schedule_generated_at
                if isinstance(generated_ts, pd.Timestamp):
                    generated_ts = generated_ts.tz_localize("UTC") if generated_ts.tzinfo is None else generated_ts.tz_convert("UTC")
                st.caption(f"Schedule generated at {generated_ts.strftime('%Y-%m-%d %H:%M UTC')}")

            if filtered_df.empty:
                st.info("No schedule entries available for the selected filters.")
            else:
                completed_games = filtered_df[filtered_df['isCompleted']]
                wins = int((completed_games['result'] == 'W').sum())
                losses = int((completed_games['result'] == 'L').sum())
                ties = int((completed_games['result'] == 'T').sum())
                record_label = f"{wins}-{losses}" + (f"-{ties}" if ties else "")
                completed_count = len(completed_games)
                points_for = float(completed_games['teamScore'].sum(skipna=True))
                points_against = float(completed_games['opponentScore'].sum(skipna=True))
                avg_points = points_for / completed_count if completed_count else 0.0
                point_diff = points_for - points_against

                metric_cols = st.columns(4)
                metric_cols[0].metric("Record", record_label if completed_count else "0-0", help="W-L-T for completed matchups")
                metric_cols[1].metric("Avg Points For", f"{avg_points:.1f}" if completed_count else "—")
                metric_cols[2].metric("Total Points For", f"{points_for:.1f}")
                metric_cols[3].metric("Point Differential", f"{point_diff:+.1f}" if completed_count else "0.0")

                filtered_df.sort_values(['startDatetime', 'matchupPeriod'], inplace=True, ignore_index=True)

                plot_matchup_scores_by_period(filtered_df, selected_team_name, st)

                def format_schedule_table(source: pd.DataFrame) -> pd.DataFrame:
                    if source is None or source.empty:
                        return pd.DataFrame()
                    return pd.DataFrame({
                        "Period": source['matchupPeriod'].astype('Int64'),
                        "Date": source['startDateLabel'].fillna("TBD"),
                        "Opponent": source['opponentLabel'].fillna("TBD"),
                        "Venue": source['homeAwayLabel'],
                        "Type": source['isPlayoffLabel'],
                        "Result": source['result'].where(source['isCompleted'], 'TBD'),
                        "Score": source['scoreline'].replace('', pd.NA),
                        "Status": source['statusText'].fillna("")
                    })

                recent_df = filtered_df[filtered_df['isCompleted']].copy()
                recent_df.sort_values('startDatetime', ascending=False, inplace=True, ignore_index=True)
                st.subheader("Recent Results")
                recent_table = format_schedule_table(recent_df.head(5))
                if recent_table.empty:
                    st.info("No matchups have been completed yet.")
                else:
                    st.dataframe(recent_table, use_container_width=True, hide_index=True)

                with st.expander("Full Schedule"):
                    full_table = format_schedule_table(filtered_df)
                    st.dataframe(full_table, use_container_width=True, hide_index=True)
