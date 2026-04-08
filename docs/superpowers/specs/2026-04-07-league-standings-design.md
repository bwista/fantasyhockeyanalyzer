# League Standings Tab — Design Spec

## Overview

Add a new "Standings" tab to the Fantasy Hockey dashboard showing league-wide standings with an all-play record. This provides a league-level view that reveals both actual performance and schedule luck.

## Requirements

- New tab positioned second: `[Draft] [Standings] [Team]`
- Flat league table (no division groupings)
- Toggle to include/exclude playoff matchups (default: exclude)
- All data sourced from existing `schedule_df` — no new API calls or data files

## Standings Table

| Column | Description | Source |
|--------|-------------|--------|
| Team | Team name | `teamName` |
| W | Wins | Count `result == 'W'` from completed matchups |
| L | Losses | Count `result == 'L'` from completed matchups |
| T | Ties | Count `result == 'T'` from completed matchups |
| PF | Points For | Sum of `teamScore` |
| PA | Points Against | Sum of `opponentScore` |
| Diff | Point Differential | PF - PA |
| All-Play | All-Play Record | See calculation below |

**Sorting:** Primary by W descending, tiebreaker by PF descending.

## All-Play Calculation

For each completed matchup period:
1. Collect every team's `teamScore` for that period
2. For each team, count how many other teams scored less (all-play win), more (all-play loss), or the same (all-play tie)
3. Aggregate across all periods

Displayed as a "W-L" string (e.g., "152-34") or "W-L-T" if ties exist.

## Layout (top to bottom)

1. **Header:** "League Standings"
2. **Toggle:** Checkbox "Include playoff matchups" (default off)
3. **Metric cards:** 3x `st.metric` — league leader in PF, best All-Play record, biggest point differential
4. **Standings table:** `st.dataframe` with all columns

## File Changes

### `dashboard_logic.py`
- `compute_standings(schedule_df: pd.DataFrame) -> pd.DataFrame` — aggregates W-L-T, PF, PA, Diff per team
- `compute_all_play_record(schedule_df: pd.DataFrame) -> pd.DataFrame` — computes all-play W-L per team across matchup periods

### `dashboard.py`
1. Hoist data loading (`ensure_data_files_exist`, freshness row, refresh button) above tab definitions so all tabs share the data
2. Add `standings_tab` between `draft_tab` and `team_tab`
3. Standings tab calls the two new logic functions and renders the layout above

### `tests/test_dashboard_logic.py`
- Tests for `compute_standings` and `compute_all_play_record` with synthetic schedule data

## No New Dependencies

All computation uses pandas operations on existing data. No new packages required.
