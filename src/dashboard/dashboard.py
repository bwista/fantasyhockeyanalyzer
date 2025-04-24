import streamlit as st
import pandas as pd

# --- Configuration ---
DRAFT_RESULTS_FILE = '../../draft_results.csv'
PLAYER_STATS_FILE = '../../player_stats.csv'
PLACEHOLDER_POINTS_COLUMN = 'G' # Use 'G' (Goals) as a temporary placeholder for points
N_PICKS_DISPLAY = 10 # Number of best/worst picks to show

# --- Helper Functions ---
def load_data(draft_file, stats_file):
    """Loads draft results and player stats from CSV files."""
    try:
        draft_df = pd.read_csv(draft_file)
        st.success(f"Loaded draft data from {draft_file}")
    except FileNotFoundError:
        st.error(f"Error: Draft results file not found at {draft_file}. Please run parse_draft_results.py.")
        return None, None
    except Exception as e:
        st.error(f"Error loading {draft_file}: {e}")
        return None, None

    try:
        stats_df = pd.read_csv(stats_file)
        st.success(f"Loaded player stats from {stats_file}")
    except FileNotFoundError:
        st.error(f"Error: Player stats file not found at {stats_file}. Please run fetch_player_stats.py.")
        return draft_df, None # Return draft_df even if stats are missing
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
        st.error(f"Error: Placeholder points column '{points_col}' not found in merged data. Cannot calculate value.")
        # Add a dummy points column to prevent crashing downstream? Or return None?
        # For now, let's add a dummy column if it's missing.
        merged_df[points_col] = 0
        st.warning(f"Added dummy '{points_col}' column with zeros.")
        # return None # Option to halt if points column is critical and missing

    # Fill NaN points with 0
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
            # Use left merge to keep all drafted players, even if they lack stats
            merged_df = pd.merge(draft_df, stats_df, on='Player', how='left')
            st.success("Successfully merged draft data and player stats.")
            # Fill NaN for stat columns that might result from the left merge
            stat_cols = stats_df.columns.difference(['Player'])
            merged_df[stat_cols] = merged_df[stat_cols].fillna(0)

        except Exception as e:
            st.error(f"Error merging dataframes: {e}")
            merged_df = draft_df # Fallback to just draft data if merge fails
            merged_df[PLACEHOLDER_POINTS_COLUMN] = 0 # Add dummy points if merge failed
    else:
        st.warning("Player stats not loaded. Displaying draft data only.")
        merged_df = draft_df # Use only draft data if stats failed to load
        merged_df[PLACEHOLDER_POINTS_COLUMN] = 0 # Add dummy points

    # Calculate Value (using placeholder points column)
    st.markdown(f"--- \n *Value calculations are currently based on **{PLACEHOLDER_POINTS_COLUMN}** as a placeholder.*")
    value_df = calculate_value(merged_df.copy(), PLACEHOLDER_POINTS_COLUMN) # Use copy to avoid modifying merged_df

    if value_df is not None:
        # Display Overall Analysis
        st.header("Overall Draft Value Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Top {N_PICKS_DISPLAY} Best Value Picks")
            # Select relevant columns for display
            best_value_cols = ['Pick', 'Player', 'Team', PLACEHOLDER_POINTS_COLUMN, 'PointsRank', 'ValueScore']
            st.dataframe(value_df.head(N_PICKS_DISPLAY)[best_value_cols], use_container_width=True)
        with col2:
            st.subheader(f"Top {N_PICKS_DISPLAY} Worst Value Picks")
            st.dataframe(value_df.tail(N_PICKS_DISPLAY).sort_values(by='ValueScore', ascending=True)[best_value_cols], use_container_width=True)

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
        st.header("Full Draft Data with Stats and Value")
        with st.expander("Show Full Data Table"):
            # Select and reorder columns for the full table display
            all_cols_ordered = ['Pick', 'Round', 'Team', 'Player', PLACEHOLDER_POINTS_COLUMN, 'ValueScore', 'PointsRank', 'DraftRank'] + \
                               [col for col in value_df.columns if col not in ['Pick', 'Round', 'Team', 'Player', PLACEHOLDER_POINTS_COLUMN, 'ValueScore', 'PointsRank', 'DraftRank']]
            st.dataframe(value_df[all_cols_ordered], use_container_width=True)

    else:
        st.warning("Could not calculate value scores.")
        # Optionally display just the merged data without value scores
        st.header("Merged Draft and Stats Data")
        st.dataframe(merged_df)

else:
    st.warning("Draft data could not be loaded. Cannot display dashboard.") 