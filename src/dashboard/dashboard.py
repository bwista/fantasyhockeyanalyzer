import streamlit as st
import pandas as pd
import json # Added for team mapping
import time # Added for temporary messages
import plotly.express as px # Added for plotting
import numpy as np # Added for trendline calculation
import statsmodels.api as sm # Added for OLS trendline calculation

# --- Configuration ---
DRAFT_RESULTS_FILE = 'src/data/draft_results.csv' # Updated path
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
    """Loads draft results (CSV), player stats (JSON), and team mapping (JSON) from files."""
    draft_df, stats_df, team_map = None, None, None # Initialize

    # --- Load Draft Data ---
    try:
        draft_df = pd.read_csv(draft_file)
        draft_df['Overall Pick'] = draft_df.index + 1 # Add Overall Pick number
        show_temporary_message("success", f"Loaded draft data from {draft_file} and added 'Overall Pick'.", 0.25)
    except FileNotFoundError:
        st.error(f"Error: Draft results file not found at {draft_file}.")
        # Allow proceeding without draft data if other files load? For now, let's require it.
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

def calculate_value(merged_df, points_col):
    """Merges data and calculates draft value."""
    if merged_df is None or merged_df.empty:
        st.warning("Cannot calculate value: Merged DataFrame is empty.")
        return None

    # Ensure points column exists and handle missing values
    if points_col not in merged_df.columns:
        # This might happen if a drafted player had zero stats entries
        st.warning(f"Points column '{points_col}' not found after merge, likely missing stats. Setting points to 0 for these players.")
        merged_df[points_col] = 0 # Add the column with 0s if it's missing entirely
        # return None # Option to halt if points column is critical and missing

    # Fill NaN points with 0 (for players drafted but without stats after merge)
    merged_df[points_col] = pd.to_numeric(merged_df[points_col], errors='coerce').fillna(0)

    # Calculate Points Rank (higher points = better rank, handle ties)
    # Using 'dense' ranking: players with same points get same rank, next rank is incremented
    merged_df['PointsRank'] = merged_df[points_col].rank(method='dense', ascending=False)

    # Draft Rank is the overall pick number
    merged_df['DraftRank'] = merged_df['Overall Pick'] # Use the new 'Overall Pick' column

    # Calculate Value Score (Higher score = better value)
    # Player drafted later (high DraftRank) but ranked high in points (low PointsRank) gets high score
    merged_df['ValueScore'] = merged_df['DraftRank'] - merged_df['PointsRank']

    # Sort by ValueScore for display
    merged_df = merged_df.sort_values(by='ValueScore', ascending=False)

    return merged_df

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title('🏒 Fantasy Hockey Draft Analysis')

# Load Data
draft_df, stats_df, team_map = load_data(DRAFT_RESULTS_FILE, PLAYER_STATS_FILE, TEAM_MAPPING_FILE)

# Proceed only if draft data is loaded
if draft_df is not None:
    # --- Data Processing & Merging ---
    processed_df = draft_df.copy() # Start with draft data

    # Initialize columns that will be added
    processed_df[POINTS_COLUMN] = 0
    processed_df['LastTeamAbbrev'] = None
    processed_df['DraftingTeamAbbrev'] = 'N/A' # Default if mapping fails
    processed_df['TeamStatus'] = 'Unknown' # Default status

    # 1. Add Drafting Team Abbreviation using Mapping (Needed before filtering stats)
    if team_map is not None:
        show_temporary_message("info", "Mapping drafting team names to abbreviations...", 0.25)
        processed_df['DraftingTeamAbbrev'] = processed_df['Team'].map(team_map).fillna('UNKNOWN')
        # Check if any mappings failed
        if 'UNKNOWN' in processed_df['DraftingTeamAbbrev'].unique():
            st.warning("Some drafting team names could not be mapped to abbreviations. Check team_mapping.json.")
        else:
            show_temporary_message("success", "Added drafting team abbreviations.", 0.25)
    else:
        st.warning("Team mapping file not found or invalid. Cannot determine drafting team abbreviations.")
        # DraftingTeamAbbrev remains 'N/A'

    # 2. Process Stats (Filter by Drafting Team, Aggregate Points, Find Last Team)
    if stats_df is not None and not stats_df.empty:
        show_temporary_message("info", "Processing player stats...", 0.25)
        try:
            # Create a lookup map for player -> drafting abbrev
            # Ensure 'Player' column exists and handle potential errors if it doesn't
            if 'Player' in processed_df.columns and 'DraftingTeamAbbrev' in processed_df.columns:
                 player_to_draft_abbrev = processed_df.set_index('Player')['DraftingTeamAbbrev'].to_dict()
                 show_temporary_message("success", "Created player-to-drafting-team lookup.", 0.25)
            else:
                 st.error("Critical columns ('Player', 'DraftingTeamAbbrev') missing in processed_df. Cannot proceed with stats filtering.")
                 player_to_draft_abbrev = {} # Empty dict to prevent further errors


            # Filter stats_df to only include points scored for the drafting team
            if player_to_draft_abbrev and 'name' in stats_df.columns and 'team_abbrev' in stats_df.columns:
                show_temporary_message("info", "Filtering weekly stats for points scored with drafting team...", 0.25)

                # Define filter condition
                def is_drafting_team(row):
                    player_name = row['name']
                    week_abbrev = row['team_abbrev']
                    draft_abbrev = player_to_draft_abbrev.get(player_name)
                    # Only count if draft abbrev is known and matches week abbrev
                    return draft_abbrev not in [None, 'N/A', 'UNKNOWN'] and week_abbrev == draft_abbrev

                filtered_stats_df = stats_df[stats_df.apply(is_drafting_team, axis=1)].copy()
                show_temporary_message("success", f"Filtered weekly stats. Kept {len(filtered_stats_df)} entries scored for drafting teams.", 0.25)

                # Aggregate points *from the filtered stats*
                if not filtered_stats_df.empty and 'total_points' in filtered_stats_df.columns:
                    show_temporary_message("info", f"Aggregating points scored for drafting team...", 0.25)
                    aggregated_draft_team_stats = filtered_stats_df.groupby('name')['total_points'].sum().reset_index()
                    aggregated_draft_team_stats = aggregated_draft_team_stats.rename(columns={'total_points': POINTS_COLUMN})
                    show_temporary_message("success", "Aggregated points scored for drafting team.", 0.25)

                    # Merge these specific points into processed_df
                    # Use left merge to keep all drafted players, fill missing points with 0 later
                    processed_df = pd.merge(processed_df.drop(columns=[POINTS_COLUMN], errors='ignore'), # Drop initial 0 column
                                            aggregated_draft_team_stats,
                                            left_on='Player', right_on='name', how='left')
                    # Drop the extra 'name' column from the merge
                    if 'name' in processed_df.columns and 'Player' in processed_df.columns:
                        processed_df = processed_df.drop(columns=['name'])
                    show_temporary_message("success", "Merged drafting team points.", 0.25)
                else:
                    st.warning("No stats found matching drafting teams after filtering, or 'total_points' column missing. Points will be 0.")
                    # Ensure POINTS_COLUMN exists, even if it's all 0s (already initialized)
            else:
                 st.warning("Could not create player-to-draft-team lookup or required columns missing in stats_df. Skipping points aggregation.")
                 # Points remain 0

            # Determine Last Known Team Abbreviation (from original unfiltered stats_df)
            if 'week' in stats_df.columns and 'team_abbrev' in stats_df.columns and 'name' in stats_df.columns:
                show_temporary_message("info", "Determining last known team abbreviation (from all stats)...", 0.25)
                last_week_idx = stats_df.groupby('name')['week'].idxmax()
                player_last_team = stats_df.loc[last_week_idx, ['name', 'team_abbrev']].rename(columns={'team_abbrev': 'LastTeamAbbrev'})
                show_temporary_message("success", "Determined last known team abbreviation per player.", 0.25)
                # Merge last team info
                processed_df = pd.merge(processed_df.drop(columns=['LastTeamAbbrev'], errors='ignore'), # Drop initial None column
                                        player_last_team,
                                        left_on='Player', right_on='name', how='left')
                # Drop the extra 'name' column again
                if 'name' in processed_df.columns and 'Player' in processed_df.columns:
                     processed_df = processed_df.drop(columns=['name'])
                show_temporary_message("success", "Merged last team abbreviations.", 0.25)
            else:
                 st.warning("Required columns missing in stats_df for determining last team. Skipping.")
                 # LastTeamAbbrev remains None

        except Exception as e:
            st.error(f"Error during stats processing or merging: {e}")
            # Ensure essential columns exist even if processing failed partially
            if POINTS_COLUMN not in processed_df.columns: processed_df[POINTS_COLUMN] = 0
            if 'LastTeamAbbrev' not in processed_df.columns: processed_df['LastTeamAbbrev'] = None
            if 'DraftingTeamAbbrev' not in processed_df.columns: processed_df['DraftingTeamAbbrev'] = 'N/A'
            if 'TeamStatus' not in processed_df.columns: processed_df['TeamStatus'] = 'Error Processing'


    # Fill NaN points with 0 after merges (for players drafted but with no matching stats)
    processed_df[POINTS_COLUMN] = processed_df[POINTS_COLUMN].fillna(0)
    # LastTeamAbbrev can remain NaN if player had no stats at all

    # 3. Determine Team Status (using DraftingTeamAbbrev and LastTeamAbbrev)
    show_temporary_message("info", "Determining player team status...", 0.25)
    def get_team_status(row):
        draft_abbrev = row['DraftingTeamAbbrev']
        last_abbrev = row['LastTeamAbbrev']

        # Handle cases where mapping might have failed or stats were missing
        if pd.isna(last_abbrev) and draft_abbrev not in ['N/A', 'UNKNOWN']:
             # Drafted, mapping worked, but no stats recorded at all
             return "Drafted / No Stats Recorded"
        elif pd.isna(last_abbrev) and draft_abbrev in ['N/A', 'UNKNOWN']:
             # Drafted, mapping failed, no stats recorded
             return "Status Unknown (No Stats & Mapping Issue)"
        elif draft_abbrev in ['N/A', 'UNKNOWN']:
             # Has stats, but drafting team unknown
             return "Status Unknown (Mapping Issue)"
        elif draft_abbrev == last_abbrev:
            return "Retained"
        else: # draft_abbrev is known, last_abbrev is known, and they differ
            return "Changed Team"

    processed_df['TeamStatus'] = processed_df.apply(get_team_status, axis=1)
    show_temporary_message("success", "Determined team status.", 0.25)


    # --- Calculate Value ---
    # Pass the fully processed DataFrame to calculate_value
    value_df = calculate_value(processed_df.copy(), POINTS_COLUMN) # Use processed_df

    # --- Display Results ---
    if value_df is not None:
        st.header("Draft Analysis Results")

        # --- Scatter Plot: Overall Pick vs Points Rank ---
        st.subheader("Overall Pick vs. Fantasy Points Rank")

        # Use the full dataset for the plot initially (filtering via legend)
        plot_df = value_df

        # --- Calculate Trendline on ALL data ---
        trendline_results = None
        trendline_x = None
        trendline_y_pred = None
        if not value_df.empty and 'Overall Pick' in value_df.columns and 'PointsRank' in value_df.columns:
            # Prepare data for OLS (handle potential NaNs just in case)
            ols_data = value_df[['Overall Pick', 'PointsRank']].dropna()
            if not ols_data.empty:
                X = ols_data['Overall Pick']
                y = ols_data['PointsRank']
                X_with_const = sm.add_constant(X)
                model = sm.OLS(y, X_with_const)
                trendline_results = model.fit()
                # Generate points for the trendline trace spanning the x-axis
                trendline_x = np.linspace(X.min(), X.max(), 100)
                trendline_x_with_const = sm.add_constant(trendline_x)
                trendline_y_pred = trendline_results.predict(trendline_x_with_const)

        # --- Generate Plot ---
        if not plot_df.empty:
            # Determine axis limits (based on full dataset for consistency)
            max_pick = value_df['Overall Pick'].max() if not value_df.empty else 160
            max_rank = value_df['PointsRank'].max() if not value_df.empty else 160
            axis_limit = max(max_pick, max_rank) * 1.2

            # Create scatter plot WITHOUT automatic trendline
            fig = px.scatter(
                plot_df, # Use filtered data for points
                x='Overall Pick',
                y='PointsRank',
                color='Team',
                hover_data=['Player', 'Overall Pick', 'PointsRank', 'Team', POINTS_COLUMN, 'ValueScore'],
                trendline=None, # Trendline will be added manually
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            # Add the pre-calculated "All Teams" trendline if available
            if trendline_x is not None and trendline_y_pred is not None:
                import plotly.graph_objects as go # Import needed for go.Scatter
                fig.add_trace(go.Scatter(
                    x=trendline_x,
                    y=trendline_y_pred,
                    mode='lines',
                    name='Overall Trend (All Teams)',
                    line=dict(color='rgba(255,255,255,0.6)', dash='dash') # Style the trendline
                ))

            # Update layout: Set fixed axes, invert Y axis
            fig.update_layout(
                xaxis_title="Overall Pick",
                yaxis_title="Points Rank (Lower is Better)",
                xaxis_range=[0, axis_limit],
                yaxis_range=[axis_limit, 0], # Inverted Y axis with fixed limit
                yaxis_autorange=False, # Disable autorange for Y
                xaxis_autorange=False  # Disable autorange for X
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Overall Value Tables ---
        st.subheader("Overall Draft Value Analysis")
        col1, col2 = st.columns(2)
        # Define columns for concise display tables (Removed status columns)
        display_cols = ['Overall Pick', 'Player', 'Team', POINTS_COLUMN, 'PointsRank', 'ValueScore']

        with col1:
            st.markdown(f"**Top {N_PICKS_DISPLAY} Best Value Picks**")
            st.dataframe(value_df.head(N_PICKS_DISPLAY)[display_cols], use_container_width=True, hide_index=True) # Use value_df
        with col2:
            st.markdown(f"**Top {N_PICKS_DISPLAY} Worst Value Picks**")
            st.dataframe(value_df.tail(N_PICKS_DISPLAY).sort_values(by='ValueScore', ascending=True)[display_cols], use_container_width=True, hide_index=True) # Use value_df

        # Display Per-Team Analysis (using original value_df)
        st.subheader("Team-Specific Draft Value Analysis")
        # Get teams from the original value_df
        teams = sorted(value_df['Team'].unique())
        if teams:
             selected_team = st.selectbox('Select Team:', teams)
             if selected_team:
                 team_df = value_df[value_df['Team'] == selected_team].copy() # Filter original value_df
                 st.markdown(f"**Analysis for {selected_team}**")

                 col3, col4 = st.columns(2)
                 with col3:
                     st.markdown(f"*Top {N_PICKS_DISPLAY} Best Value Picks*")
                     st.dataframe(team_df.head(N_PICKS_DISPLAY)[display_cols], use_container_width=True, hide_index=True)
                 with col4:
                     st.markdown(f"*Top {N_PICKS_DISPLAY} Worst Value Picks*")
                     st.dataframe(team_df.tail(N_PICKS_DISPLAY).sort_values(by='ValueScore', ascending=True)[display_cols], use_container_width=True, hide_index=True)
        else:
            show_temporary_message("info", "No teams found in the data.", 0.25) # Simplified message


        # Display Full Data (Optional, using original value_df)
        st.subheader("Full Processed Data")
        with st.expander("Show Full Data Table"):
            # Define columns for the full table display (Removed status columns)
            all_cols_ordered = [
                'Overall Pick', 'Round', 'Pick', 'Team', 'Player', # Draft Info
                # 'DraftingTeamAbbrev', 'LastTeamAbbrev', 'TeamStatus', # Team Status Info (REMOVED FROM DISPLAY)
                POINTS_COLUMN, 'PointsRank', 'DraftRank', 'ValueScore' # Value Info
            ]
            # Add any remaining columns automatically, excluding the ones we explicitly removed from display
            # Note: The status columns still *exist* in value_df, just not shown here.
            explicitly_hidden = ['DraftingTeamAbbrev', 'LastTeamAbbrev', 'TeamStatus']
            remaining_cols = [col for col in value_df.columns if col not in all_cols_ordered and col not in explicitly_hidden]
            # Display using the original value_df
            st.dataframe(value_df[all_cols_ordered + remaining_cols], use_container_width=True, hide_index=True)

    else:
        st.warning("Could not calculate value scores.")
        # Display the processed data even if value calculation failed
        st.header("Processed Draft Data (Before Value Calculation)")
        st.dataframe(processed_df, hide_index=True)

else:
    st.warning("Draft data could not be loaded. Cannot display dashboard.")
