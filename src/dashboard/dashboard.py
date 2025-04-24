import streamlit as st
import pandas as pd

# --- Configuration ---
DRAFT_RESULTS_FILE = 'src/data/draft_results.csv' # Updated path
PLAYER_STATS_FILE = 'src/data/box_score_stats.json' # Updated path and extension
POINTS_COLUMN = 'TotalPoints' # Actual points column name after aggregation
STATS_RAW_POINTS_COLUMN = 'total_points' # Column name in the raw JSON stats file
N_PICKS_DISPLAY = 10 # Number of best/worst picks to show

# --- Helper Functions ---
def load_data(draft_file, stats_file):
    """Loads draft results (CSV) and player stats (JSON) from files."""
    try:
        draft_df = pd.read_csv(draft_file)
        st.success(f"Loaded draft data from {draft_file}")
    except FileNotFoundError:
        st.error(f"Error: Draft results file not found at {draft_file}. Please ensure it exists.") # Updated error message
        return None, None
    except Exception as e:
        st.error(f"Error loading {draft_file}: {e}")
        return None, None

    try:
        # Load stats from JSON
        stats_df = pd.read_json(stats_file) # Changed to read_json
        st.success(f"Loaded player stats from {stats_file}")
    except FileNotFoundError:
        st.error(f"Error: Player stats file not found at {stats_file}. Please ensure it exists.") # Updated error message
        return draft_df, None # Return draft_df even if stats are missing
    except ValueError as e: # Catch JSON decoding errors
        st.error(f"Error parsing JSON file {stats_file}: {e}")
        return draft_df, None
    except Exception as e:
        st.error(f"Error loading {stats_file}: {e}")
        return draft_df, None

    return draft_df, stats_df

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

    # Draft Rank is simply the pick number
    merged_df['DraftRank'] = merged_df['Pick'] # Assuming 'Pick' column exists from draft_results

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
draft_df, stats_df = load_data(DRAFT_RESULTS_FILE, PLAYER_STATS_FILE)

# Proceed only if draft data is loaded
if draft_df is not None:
    # Merge DataFrames
    merged_df = None
    if stats_df is not None:
        try:
            # Aggregate weekly stats to get total points per player
            if STATS_RAW_POINTS_COLUMN in stats_df.columns and 'Player' in stats_df.columns:
                st.info(f"Aggregating player stats by summing '{STATS_RAW_POINTS_COLUMN}'...")
                # Group by Player and sum the raw points column
                aggregated_stats = stats_df.groupby('Player')[STATS_RAW_POINTS_COLUMN].sum().reset_index()
                # Rename the summed column to the final POINTS_COLUMN name
                aggregated_stats = aggregated_stats.rename(columns={STATS_RAW_POINTS_COLUMN: POINTS_COLUMN})
                st.success("Player stats aggregated successfully.")

                # Use left merge to keep all drafted players, even if they lack stats
                merged_df = pd.merge(draft_df, aggregated_stats, on='Player', how='left')
                st.success("Successfully merged draft data and aggregated player stats.")

                # Fill NaN for the points column resulting from the left merge (drafted players with no stats)
                # Other stat columns are not present anymore after aggregation
                merged_df[POINTS_COLUMN] = merged_df[POINTS_COLUMN].fillna(0)
            else:
                st.error(f"Required columns ('Player', '{STATS_RAW_POINTS_COLUMN}') not found in stats data. Cannot aggregate.")
                merged_df = draft_df # Fallback to just draft data
                merged_df[POINTS_COLUMN] = 0 # Add dummy points column

        except Exception as e:
            st.error(f"Error aggregating or merging dataframes: {e}")
            merged_df = draft_df # Fallback to just draft data if merge fails
            merged_df[POINTS_COLUMN] = 0 # Add dummy points if merge failed
    else:
        st.warning("Player stats not loaded. Displaying draft data only.")
        merged_df = draft_df # Use only draft data if stats failed to load
        merged_df[POINTS_COLUMN] = 0 # Add dummy points

    # Calculate Value (using actual points column)
    st.markdown(f"--- \n *Value calculations based on **{POINTS_COLUMN}**.*") # Updated text
    value_df = calculate_value(merged_df.copy(), POINTS_COLUMN) # Use actual points column

    if value_df is not None:
        # Display Overall Analysis
        st.header("Overall Draft Value Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Top {N_PICKS_DISPLAY} Best Value Picks")
            # Select relevant columns for display
            best_value_cols = ['Pick', 'Player', 'Team', POINTS_COLUMN, 'PointsRank', 'ValueScore'] # Use actual points column
            st.dataframe(value_df.head(N_PICKS_DISPLAY)[best_value_cols], use_container_width=True)
        with col2:
            st.subheader(f"Top {N_PICKS_DISPLAY} Worst Value Picks")
            st.dataframe(value_df.tail(N_PICKS_DISPLAY).sort_values(by='ValueScore', ascending=True)[best_value_cols], use_container_width=True) # Use actual points column

        # Display Per-Team Analysis
        st.header("Team-Specific Draft Value Analysis")
        teams = sorted(value_df['Team'].unique())
        selected_team = st.selectbox('Select Team:', teams)

        if selected_team:
            team_df = value_df[value_df['Team'] == selected_team].copy()
            st.subheader(f"Analysis for {selected_team}")

            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"**Top {N_PICKS_DISPLAY} Best Value Picks**")
                st.dataframe(team_df.head(N_PICKS_DISPLAY)[best_value_cols], use_container_width=True)
            with col4:
                st.markdown(f"**Top {N_PICKS_DISPLAY} Worst Value Picks**")
                st.dataframe(team_df.tail(N_PICKS_DISPLAY).sort_values(by='ValueScore', ascending=True)[best_value_cols], use_container_width=True)

        # Display Full Data (Optional)
        st.header("Full Draft Data with Value Calculation")
        with st.expander("Show Full Data Table"):
            # Select and reorder columns for the full table display
            # Note: Only aggregated points are available now, not detailed weekly stats
            all_cols_ordered = ['Pick', 'Round', 'Team', 'Player', POINTS_COLUMN, 'ValueScore', 'PointsRank', 'DraftRank'] + \
                               [col for col in value_df.columns if col not in ['Pick', 'Round', 'Team', 'Player', POINTS_COLUMN, 'ValueScore', 'PointsRank', 'DraftRank']]
            st.dataframe(value_df[all_cols_ordered], use_container_width=True) # Use actual points column

    else:
        st.warning("Could not calculate value scores.")
        # Optionally display just the merged data without value scores
        st.header("Merged Draft and Stats Data")
        st.dataframe(merged_df)

else:
    st.warning("Draft data could not be loaded. Cannot display dashboard.")
