"""
Core logic for the Fantasy Hockey Dashboard: data loading, processing, and plotting functions.
"""
import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# --- Data Loading/Generation ---
def ensure_data_files_exist(config, draft_results_file, player_stats_file, team_mapping_file,
                           parse_draft_results, fetch_box_score_stats, fetch_and_save_team_info,
                           start_week, end_week, rate_limit_delay, team_info_output_dir, team_info_output_file, st=None):
    """
    Ensures all required data files exist, generating them if necessary. Returns loaded dataframes and mapping.
    """
    league_id = config.get('LEAGUE_ID')
    year = config.get('YEAR')
    swid = config.get('SWID')
    espn_s2 = config.get('ESPN_S2')
    DATA_DIR = os.path.dirname(draft_results_file)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        if st: st.info(f"Created data directory: {DATA_DIR}")

    # Draft Results
    if not os.path.exists(draft_results_file):
        if st: st.warning(f"Draft results file ({draft_results_file}) not found. Attempting to generate...")
        draft_data = parse_draft_results(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)
        if draft_data is not None:
            draft_data.to_json(draft_results_file, orient='records', indent=4)
            if st: st.success(f"Generated and saved draft results to {draft_results_file}.")
        else:
            if st: st.error("Failed to generate draft results.")

    # Player Stats
    if not os.path.exists(player_stats_file):
        if st: st.warning(f"Player stats file ({player_stats_file}) not found. Attempting to generate...")
        box_stats = fetch_box_score_stats(
            league_id=league_id, year=year, swid=swid, espn_s2=espn_s2,
            start_week=start_week, end_week=end_week, delay=rate_limit_delay
        )
        if box_stats:
            with open(player_stats_file, 'w') as f:
                json.dump(box_stats, f, indent=4)
            if st: st.success(f"Generated and saved player stats to {player_stats_file}.")
        elif box_stats == []:
            if st: st.warning("Player stats generation function returned no data.")

    # Team Mapping
    if not os.path.exists(team_mapping_file):
        if st: st.warning(f"Team mapping file ({team_mapping_file}) not found. Attempting to generate...")
        success = fetch_and_save_team_info(
            league_id=league_id, year=year, swid=swid, espn_s2=espn_s2,
            output_dir=team_info_output_dir, output_file=team_info_output_file
        )
        if success and os.path.exists(team_mapping_file):
            if st: st.success(f"Generated and saved team mapping to {team_mapping_file}.")
        else:
            if st: st.error("Failed to generate team mapping.")

    # Load data
    draft_df, stats_df, team_map = load_data(draft_results_file, player_stats_file, team_mapping_file, st=st)
    return draft_df, stats_df, team_map


def load_data(draft_file, stats_file, mapping_file, st=None):
    """
    Loads draft results (JSON), player stats (JSON), and team mapping (JSON) from files.
    """
    draft_df, stats_df, team_map = None, None, None
    try:
        draft_df = pd.read_json(draft_file, orient='records')
        draft_df['DraftPick'] = draft_df.index + 1
        if st: st.success(f"Loaded draft data from {draft_file} (JSON) and added 'Pick'.")
    except Exception as e:
        if st: st.error(f"Error loading {draft_file}: {e}")
        return None, None, None
    try:
        stats_df = pd.read_json(stats_file)
        if st: st.success(f"Loaded player stats from {stats_file}")
    except Exception as e:
        if st: st.warning(f"Warning: Could not load player stats from {stats_file}: {e}")
    try:
        with open(mapping_file, 'r') as f:
            team_map = json.load(f)
        if st: st.success(f"Loaded team mapping from {mapping_file}")
    except Exception as e:
        if st: st.warning(f"Warning: Could not load team mapping from {mapping_file}: {e}")
    return draft_df, stats_df, team_map

# --- Data Processing ---
def process_data(draft_df, stats_df, team_map, st=None):
    """
    Processes and merges draft, stats, and team mapping data. Returns final_df and value_df.
    """
    if draft_df is None or stats_df is None:
        if st: st.warning("Draft or stats data missing.")
        return None, None
    try:
        # Aggregate per player/team
        player_team_stats = stats_df.groupby(['name', 'team_abbrev', 'position']).agg(
            TeamPoints=('total_points', 'sum'),
            FirstWeek=('week', 'min'),
            LastWeek=('week', 'max')
        ).reset_index()
        # Map position values to F, D, G
        position_map = {
            'Left Wing': 'F',
            'Right Wing': 'F',
            'Center': 'F',
            'Defense': 'D',
            'Goalie': 'G'
        }
        player_team_stats['position'] = player_team_stats['position'].replace(position_map)
        # Aggregate overall player stats
        total_player_stats = stats_df.groupby('name').agg(
            TotalPoints=('total_points', 'sum')
        ).reset_index()
        # Prepare draft data
        draft_prepared_df = draft_df[['Player', 'DraftPick', 'Team']].copy()
        draft_prepared_df.rename(columns={'Player': 'name', 'Team': 'DraftingTeamName'}, inplace=True)
        if team_map is not None:
            draft_prepared_df['DraftingTeamAbbrev'] = draft_prepared_df['DraftingTeamName'].map(team_map)
            unmapped_mask = draft_prepared_df['DraftingTeamAbbrev'].isna()
            draft_prepared_df.loc[unmapped_mask, 'DraftingTeamAbbrev'] = 'UNMAPPED_ABBREV'
        else:
            draft_prepared_df['DraftingTeamAbbrev'] = 'NO_TEAM_MAP_ABBREV'
        # Merge draft info
        merged_team_stats = pd.merge(
            player_team_stats,
            draft_prepared_df[['name', 'DraftPick', 'DraftingTeamAbbrev', 'DraftingTeamName']],
            on='name', how='left'
        )
        # Acquisition type
        final_team_stats_df = merged_team_stats.groupby('name', group_keys=False).apply(determine_acquisition_type)
        # Merge with overall stats
        final_df = pd.merge(
            final_team_stats_df,
            total_player_stats,
            on='name', how='left'
        )
        if team_map is not None:
            # Reverse the team_map: abbreviation -> team name
            abbrev_to_team = {abbrev: name for name, abbrev in team_map.items()}
            final_df['PickupTeamName'] = final_df['team_abbrev'].map(abbrev_to_team)
            unmapped_mask = final_df['PickupTeamName'].isna()
            final_df.loc[unmapped_mask, 'PickupTeamName'] = 'UNMAPPED_ABBREV'
        else:
            final_df['PickupTeamName'] = 'NO_TEAM_MAP_ABBREV'
        final_df['TotalPoints'] = final_df['TotalPoints'].fillna(0)
        value_df = calculate_value(final_df.copy(), 'TeamPoints') # Use TeamPoints for ValueScore calculation
        final_df.rename(columns={'name':'Player', 'DraftPick':'Pick', 'position':'Pos', 'DraftingTeamName':'Team'}, inplace=True)
        value_df.rename(columns={'name':'Player', 'DraftPick':'Pick', 'position':'Pos', 'DraftingTeamName':'Team'}, inplace=True)
        return final_df, value_df
    except Exception as e:
        if st: st.error(f"Error during data processing: {e}")
        return None, None

def determine_acquisition_type(group):
    group = group.sort_values('FirstWeek')
    acquisition_types = []
    for i, row in group.iterrows():
        current_type = 'waiver'
        if i == group.index[0]:
            if pd.notna(row['DraftPick']) and row['team_abbrev'] == row['DraftingTeamAbbrev']:
                current_type = 'drafted'
        acquisition_types.append(current_type)
    group['acquisition_type'] = acquisition_types
    return group

def calculate_value(df, points_col):
    if df is None or df.empty:
        return None
    required_cols = [points_col, 'DraftPick', 'acquisition_type']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0 if col == points_col else np.nan
    df[points_col] = pd.to_numeric(df[points_col], errors='coerce').fillna(0)
    df['DraftPick'] = pd.to_numeric(df['DraftPick'], errors='coerce')
    df['PointsRank'] = df[points_col].rank(method='dense', ascending=False)
    df['ValueScore'] = np.nan
    drafted_mask = (df['acquisition_type'] == 'drafted') & df['DraftPick'].notna() & df['PointsRank'].notna()
    df.loc[drafted_mask, 'ValueScore'] = df.loc[drafted_mask, 'DraftPick'] - df.loc[drafted_mask, 'PointsRank']
    return df

# --- Plotting/Display Functions ---
def plot_draft_value(value_df, st):
    st.subheader("Draft Value: Pick vs. Rank(Fantasy Points)")
    st.markdown("_This plot compares drafted players' initial position to their final season rank._")
    drafted_plot_df = value_df[value_df['acquisition_type'] == 'drafted'].copy()

    available_teams = sorted(drafted_plot_df['Team'].dropna().unique())
    position_options = ['All', 'F', 'D', 'G']
    plot_col, filter_col = st.columns([5, 1], gap="medium")

    team_options = ['All Teams'] + available_teams if available_teams else ['All Teams']

    with filter_col:
        selected_team = st.selectbox(
            "Team Filter",
            team_options,
            index=0
        )
        selected_position = st.selectbox(
            "Position Filter",
            position_options,
            index=0
        )

    if selected_team != 'All Teams':
        drafted_plot_df = drafted_plot_df[drafted_plot_df['Team'] == selected_team]

    if selected_position != 'All':
        drafted_plot_df = drafted_plot_df[drafted_plot_df['Pos'] == selected_position]

    if drafted_plot_df.empty:
        with plot_col:
            st.info("No drafted players found to plot for the selected filter.")
        return

    trendline_x = trendline_y_pred = None
    if 'Pick' in drafted_plot_df.columns and 'PointsRank' in drafted_plot_df.columns:
        ols_data = drafted_plot_df[['Pick', 'PointsRank']].dropna()
        if not ols_data.empty and len(ols_data) > 1:
            X = ols_data['Pick']
            y = ols_data['PointsRank']
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const)
            trendline_results = model.fit()
            trendline_x = np.linspace(X.min(), X.max(), 100)
            trendline_x_with_const = sm.add_constant(trendline_x)
            trendline_y_pred = trendline_results.predict(trendline_x_with_const)

    max_pick = drafted_plot_df['Pick'].dropna().max()
    max_rank = drafted_plot_df['PointsRank'].dropna().max()
    x_axis_limit = (max_pick if pd.notna(max_pick) else 160) * 1.05
    y_axis_limit = (max_rank if pd.notna(max_rank) else 160) * 1.05

    position_symbols = {'F': 'circle', 'D': 'square', 'G': 'diamond-open'}

    fig = px.scatter(
        drafted_plot_df,
        x='Pick',
        y='PointsRank',
        color='Team',
        hover_data=['Player', 'Pick', 'PointsRank', 'Team', 'TotalPoints', 'ValueScore', 'Pos'],
        color_discrete_sequence=px.colors.qualitative.Bold,
        custom_data=['Pos']
    )

    team_colors = {}

    for trace in fig.data:
        if getattr(trace, "customdata", None) is not None and trace.mode == 'markers':
            customdata_list = []
            for entry in trace.customdata:
                if isinstance(entry, (list, tuple, np.ndarray)):
                    customdata_list.append(list(entry))
                else:
                    customdata_list.append([entry])
            symbols = [
                position_symbols.get(point_data[0], 'circle')
                for point_data in customdata_list
            ]
            if selected_position == 'All':
                x_values = list(trace.x)
                y_values = list(trace.y)
                x_values.insert(0, None)
                y_values.insert(0, None)
                customdata_list.insert(0, [''])
                symbols.insert(0, 'circle')
                trace.update(x=x_values, y=y_values, customdata=customdata_list, marker=dict(symbol=symbols))
            else:
                trace.update(customdata=customdata_list, marker=dict(symbol=symbols))

            color_values = trace.marker.color
            if isinstance(color_values, (list, tuple, np.ndarray)) and len(color_values) > 0:
                team_color = color_values[0]
            else:
                team_color = color_values
            team_colors[trace.name] = team_color
            trace.showlegend = False

    for team, color in team_colors.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(color=color, symbol='circle', size=10),
            name=team,
            showlegend=True
        ))

    if trendline_x is not None and trendline_y_pred is not None:
        fig.add_trace(go.Scatter(
            x=trendline_x,
            y=trendline_y_pred,
            mode='lines',
            name='Trend (Drafted Players)',
            line=dict(color='rgba(255,255,255,0.6)', dash='dash')
        ))

    for pos, symbol in position_symbols.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(symbol=symbol, color='rgba(150,150,150,0.8)', size=10),
            name=f"Pos: {pos}",
            showlegend=True
        ))

    fig.update_layout(
        xaxis_title="Draft Pick",
        yaxis_title="Points Rank",
        xaxis_range=[0, x_axis_limit],
        yaxis_range=[0, y_axis_limit],
        yaxis_autorange=False,
        xaxis_autorange=False,
        legend_title_text='Team / Position'
    )

    with plot_col:
        st.plotly_chart(fig, use_container_width=True)

# Additional display/plotting functions can be added here as needed.
