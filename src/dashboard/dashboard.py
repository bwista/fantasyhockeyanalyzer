import streamlit as st
import pandas as pd
import json # Added for team mapping
import time # Added for temporary messages
import plotly.express as px # Added for plotting
import numpy as np # Added for trendline calculation
import statsmodels.api as sm # Added for OLS trendline calculation

# --- Configuration ---
DRAFT_RESULTS_FILE = 'src/data/draft_results.json' # Updated path for JSON
PLAYER_STATS_FILE = 'src/data/box_score_stats.json' # Updated path and extension
TEAM_MAPPING_FILE = 'src/data/team_mapping.json' # Added path for team name to abbreviation mapping
POINTS_COLUMN = 'TotalPoints' # Actual points column name after aggregation
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
        draft_df['Overall Pick'] = draft_df.index + 1
        show_temporary_message("success", f"Loaded draft data from {draft_file} (JSON) and added 'Overall Pick'.", 0.25)
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
    required_cols = [points_col, 'Overall Pick', 'acquisition_type']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Cannot calculate value: Missing one or more required columns: {required_cols}")
        # Add missing columns with default values if possible, or return None
        if points_col not in df.columns: df[points_col] = 0
        if 'Overall Pick' not in df.columns: df['Overall Pick'] = np.nan
        if 'acquisition_type' not in df.columns: df['acquisition_type'] = 'unknown'
        # Consider returning None if critical columns are missing and cannot be defaulted reasonably
        # return None

    # Ensure points column is numeric and fill NaNs
    df[points_col] = pd.to_numeric(df[points_col], errors='coerce').fillna(0)
    # Ensure Overall Pick is numeric (can be NaN for non-drafted)
    df['Overall Pick'] = pd.to_numeric(df['Overall Pick'], errors='coerce')


    # Calculate Points Rank based on overall TotalPoints (higher points = better rank)
    df['PointsRank'] = df[points_col].rank(method='dense', ascending=False)

    # Draft Rank is the overall pick number (will be NaN for non-drafted)
    df['DraftRank'] = df['Overall Pick']

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
            if pd.notna(row['Overall Pick']) and row['team_abbrev'] == row['DraftingTeamAbbrev']:
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
st.title('🏒 Fantasy Hockey Draft & Acquisition Analysis') # Updated Title

# Load Data
draft_df, stats_df, team_map = load_data(DRAFT_RESULTS_FILE, PLAYER_STATS_FILE, TEAM_MAPPING_FILE)

# Proceed only if draft and stats data are loaded
if draft_df is not None and stats_df is not None:
    # --- Data Processing Steps from PLAN.md ---
    final_df = None # Initialize final_df
    try:
        show_temporary_message("info", "Starting data processing...", 0.25)

        # I.2 Aggregate Stats per Player per Team
        if 'name' in stats_df.columns and 'team_abbrev' in stats_df.columns and 'total_points' in stats_df.columns and 'week' in stats_df.columns:
            show_temporary_message("info", "Aggregating stats per player/team...", 0.25)
            player_team_stats = stats_df.groupby(['name', 'team_abbrev']).agg(
                TeamPoints=('total_points', 'sum'),
                FirstWeek=('week', 'min'),
                LastWeek=('week', 'max')
            ).reset_index()
            show_temporary_message("success", f"Aggregated stats for {len(player_team_stats)} player-team combinations.", 0.25)
        else:
            st.error("Required columns ('name', 'team_abbrev', 'total_points', 'week') missing in stats_df. Cannot perform per-team aggregation.")
            st.stop() # Stop execution if basic stats aggregation fails

        # I.3 Aggregate Overall Player Stats
        if 'name' in stats_df.columns and 'total_points' in stats_df.columns:
             show_temporary_message("info", "Aggregating overall player stats...", 0.25)
             total_player_stats = stats_df.groupby('name').agg(
                 TotalPoints=('total_points', 'sum')
             ).reset_index()
             show_temporary_message("success", f"Aggregated total points for {len(total_player_stats)} unique players.", 0.25)
        else:
             st.error("Required columns ('name', 'total_points') missing in stats_df. Cannot perform overall aggregation.")
             st.stop()

        # I.4 Prepare Draft Data
        show_temporary_message("info", "Preparing draft data...", 0.25)
        draft_prepared_df = draft_df[['Player', 'Overall Pick', 'Team']].copy()
        if team_map is not None:
            draft_prepared_df['DraftingTeamAbbrev'] = draft_prepared_df['Team'].map(team_map).fillna('UNKNOWN_MAP')
            if 'UNKNOWN_MAP' in draft_prepared_df['DraftingTeamAbbrev'].unique():
                 st.warning("Some drafting team names could not be mapped. Check team_mapping.json.")
        else:
            st.warning("Team mapping not available. DraftingTeamAbbrev cannot be added.")
            draft_prepared_df['DraftingTeamAbbrev'] = 'N/A'
        draft_prepared_df = draft_prepared_df.rename(columns={'Player': 'name'})
        show_temporary_message("success", "Prepared draft data.", 0.25)

        # I.5 Merge Draft Info with Per-Team Stats
        show_temporary_message("info", "Merging draft info into per-team stats...", 0.25)
        merged_team_stats = pd.merge(
            player_team_stats,
            draft_prepared_df[['name', 'Overall Pick', 'DraftingTeamAbbrev']],
            on='name',
            how='left' # Keep all player-team stats, add draft info where available
        )
        show_temporary_message("success", "Merged draft info.", 0.25)

        # I.6 Determine Acquisition Type
        show_temporary_message("info", "Determining acquisition type for each player-team record...", 0.25)
        # Apply the helper function group-wise
        # Important: Ensure data types are compatible before grouping if necessary
        final_team_stats_df = merged_team_stats.groupby('name', group_keys=False).apply(determine_acquisition_type)
        show_temporary_message("success", "Determined acquisition types.", 0.25)

        # I.7 Combine with Overall Stats
        show_temporary_message("info", "Merging overall total points...", 0.25)
        final_df = pd.merge(
            final_team_stats_df,
            total_player_stats,
            on='name',
            how='left' # Should match all records from final_team_stats_df
        )
        # Fill NaN TotalPoints just in case (shouldn't happen with left merge if total_player_stats is complete)
        final_df['TotalPoints'] = final_df['TotalPoints'].fillna(0)
        show_temporary_message("success", "Final DataFrame created.", 0.25)

    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        st.exception(e) # Show traceback for debugging
        final_df = None # Ensure final_df is None if processing fails

    # --- Calculate Value (using the new final_df) ---
    if final_df is not None:
        value_df = calculate_value(final_df.copy(), 'TotalPoints') # Pass 'TotalPoints' as points_col
    else:
        value_df = None # Ensure value_df is None if processing failed

    # --- Display Results ---
    if value_df is not None:
        st.header("Draft & Acquisition Analysis Results") # Updated Header

        # --- Scatter Plot: Overall Pick vs Points Rank (Drafted Players Only) ---
        st.subheader("Draft Value: Overall Pick vs. Fantasy Points Rank")
        st.markdown("_This plot shows only players who were initially drafted and plots their draft position against their final points rank for the season._")

        # Filter for drafted records only for plotting draft value
        drafted_plot_df = value_df[value_df['acquisition_type'] == 'drafted'].copy()

        # --- Calculate Trendline on DRAFTED data ---
        trendline_results = None
        trendline_x = None
        trendline_y_pred = None
        if not drafted_plot_df.empty and 'Overall Pick' in drafted_plot_df.columns and 'PointsRank' in drafted_plot_df.columns:
            # Prepare data for OLS (handle potential NaNs just in case)
            ols_data = drafted_plot_df[['Overall Pick', 'PointsRank']].dropna()
            if not ols_data.empty and len(ols_data) > 1: # Need at least 2 points for trendline
                X = ols_data['Overall Pick']
                y = ols_data['PointsRank']
                X_with_const = sm.add_constant(X)
                try:
                    model = sm.OLS(y, X_with_const)
                    trendline_results = model.fit()
                    # Generate points for the trendline trace spanning the x-axis
                    trendline_x = np.linspace(X.min(), X.max(), 100)
                    trendline_x_with_const = sm.add_constant(trendline_x)
                    trendline_y_pred = trendline_results.predict(trendline_x_with_const)
                except Exception as e:
                    st.warning(f"Could not calculate trendline: {e}")


        # --- Generate Plot ---
        if not drafted_plot_df.empty:
            # Determine axis limits (based on drafted dataset)
            max_pick = drafted_plot_df['Overall Pick'].max() if not drafted_plot_df.empty else 160
            max_rank = drafted_plot_df['PointsRank'].max() if not drafted_plot_df.empty else 160
            # Use overall max rank from value_df for y-axis consistency if needed
            max_rank = value_df['PointsRank'].max() if not value_df.empty else 160
            x_axis_limit = max_pick * 1.2
            y_axis_limit = max_rank * 1.2

            # Create scatter plot WITHOUT automatic trendline
            fig = px.scatter(
                drafted_plot_df, # Use filtered drafted data
                x='Overall Pick',
                y='PointsRank',
                color='DraftingTeamAbbrev', # Color by drafting team
                hover_data=['name', 'Overall Pick', 'PointsRank', 'DraftingTeamAbbrev', 'TotalPoints', 'ValueScore'], # Use name, TotalPoints
                title="Draft Position vs. Season Points Rank (Drafted Players)",
                trendline=None, # Trendline will be added manually
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            # Add the pre-calculated "Drafted Players" trendline if available
            if trendline_x is not None and trendline_y_pred is not None:
                import plotly.graph_objects as go # Import needed for go.Scatter
                fig.add_trace(go.Scatter(
                    x=trendline_x,
                    y=trendline_y_pred,
                    mode='lines',
                    name='Trend (Drafted Players)',
                    line=dict(color='rgba(255,255,255,0.6)', dash='dash') # Style the trendline
                ))

            # Update layout: Set fixed axes, invert Y axis
            fig.update_layout(
                xaxis_title="Overall Pick",
                yaxis_title="Points Rank (Lower is Better)",
                xaxis_range=[0, x_axis_limit],
                yaxis_range=[y_axis_limit, 0], # Inverted Y axis with fixed limit
                yaxis_autorange=False, # Disable autorange for Y
                xaxis_autorange=False,  # Disable autorange for X
                legend_title_text='Drafting Team'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("No 'drafted' players found to plot.")

        # --- Overall Value Tables (Based on Drafted Players) ---
        st.subheader("Overall Draft Value Analysis")
        st.markdown("_Based on players' performance relative to their draft position (`ValueScore = DraftRank - PointsRank`). Higher is better._")

        # Filter for drafted records for value tables
        drafted_value_df = value_df[value_df['acquisition_type'] == 'drafted'].sort_values(by='ValueScore', ascending=False)

        col1, col2 = st.columns(2)
        # Define columns for concise display tables
        display_cols_value = ['Overall Pick', 'name', 'DraftingTeamAbbrev', 'TotalPoints', 'PointsRank', 'ValueScore']

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
                 st.info("No 'drafted' players found.")

        # --- Team-Specific Draft Value Analysis (Based on Drafted Players) ---
        st.subheader("Team-Specific Draft Value Analysis")
        # Get drafting teams from the drafted subset
        drafting_teams = sorted(drafted_value_df['DraftingTeamAbbrev'].dropna().unique())
        if drafting_teams:
             selected_draft_team = st.selectbox('Select Drafting Team:', drafting_teams)
             if selected_draft_team:
                 team_value_df = drafted_value_df[drafted_value_df['DraftingTeamAbbrev'] == selected_draft_team].copy()
                 st.markdown(f"**Draft Value Analysis for {selected_draft_team}**")

                 col3, col4 = st.columns(2)
                 with col3:
                     st.markdown(f"*Top {N_PICKS_DISPLAY} Best Value Picks*")
                     st.dataframe(team_value_df.head(N_PICKS_DISPLAY)[display_cols_value], use_container_width=True, hide_index=True)
                 with col4:
                     st.markdown(f"*Top {N_PICKS_DISPLAY} Worst Value Picks*")
                     st.dataframe(team_value_df.tail(N_PICKS_DISPLAY).sort_values(by='ValueScore', ascending=True)[display_cols_value], use_container_width=True, hide_index=True)
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
            # Get unique players acquired via waiver
            unique_waiver_players = waiver_records_df[['name', 'TotalPoints', 'Overall Pick', 'DraftingTeamAbbrev']].drop_duplicates(subset=['name'])
            # Sort them by TotalPoints
            top_overall_acquisitions = unique_waiver_players.sort_values(by='TotalPoints', ascending=False)
            # Display
            display_cols_overall_acq = ['name', 'TotalPoints', 'Overall Pick', 'DraftingTeamAbbrev']
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
                    # Sort by points scored for *this* team
                    team_acquisitions_df = team_acquisitions_df.sort_values(by='TeamPoints', ascending=False)
                    st.markdown(f"**Top {N_PICKS_DISPLAY} Acquisitions for {selected_acq_team} (by Points for Team)**")
                    display_cols_team_acq = ['name', 'TeamPoints', 'FirstWeek', 'LastWeek'] # Show points for team and duration
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
                'name', 'team_abbrev', 'acquisition_type', 'TeamPoints', 'TotalPoints', # Core Info
                'FirstWeek', 'LastWeek', # Stint Info
                'Overall Pick', 'DraftingTeamAbbrev', # Draft Info (if applicable)
                'PointsRank', 'ValueScore' # Ranks & Value (if applicable)
            ]
            # Add any remaining columns automatically
            remaining_cols = [col for col in value_df.columns if col not in all_cols_ordered]
            # Display using the final value_df
            st.dataframe(value_df[all_cols_ordered + remaining_cols], use_container_width=True, hide_index=True)

    else:
        st.warning("Could not process data or calculate value scores.")
        # Attempt to display intermediate data if available
        if 'final_df' in locals() and final_df is not None:
             st.header("Processed Data (Before Value Calculation)")
             st.dataframe(final_df, hide_index=True)
        elif 'merged_team_stats' in locals() and merged_team_stats is not None:
             st.header("Merged Team Stats (Before Acquisition Type)")
             st.dataframe(merged_team_stats, hide_index=True)
        else:
             st.info("No intermediate data available to display.")


elif draft_df is None:
     st.warning("Draft data could not be loaded. Cannot display dashboard.")
else: # stats_df is None
     st.warning("Stats data could not be loaded. Cannot perform analysis.") # Added specific message for missing stats
