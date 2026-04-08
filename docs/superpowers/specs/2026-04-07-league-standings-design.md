# League Standings Tab — Design Spec

## Overview

Add a new "Standings" tab to the Fantasy Hockey dashboard showing league-wide standings with an all-play record. This provides a league-level view that reveals both actual performance and schedule luck.

## Requirements

- New tab positioned second: `[Draft] [Standings] [Team]`
- Flat league table (no division groupings)
- Toggle to include/exclude playoff matchups (default: exclude)
- All data sourced from existing `schedule_df` — no new API calls or data files

## Data Shape

`schedule_df` contains one row per team per matchup period (from that team's perspective). An 8-team league with 24 periods produces 192 rows. Each matchup appears twice (once per team). Use `teamScore` directly per team row — do not merge home/away pairs or you will double-count.

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

**Sorting:** Primary by W descending, tiebreaker by PF descending, then Diff descending.

## All-Play Calculation

For each completed matchup period:
1. Collect every team's `teamScore` for that period
2. For each team, count how many other teams scored less (all-play win), more (all-play loss), or the same (all-play tie)
3. Aggregate across all periods

Displayed as a "W-L" string (e.g., "152-34") or "W-L-T" if ties exist.

## Pre-Filtering

Both logic functions receive a **pre-filtered** DataFrame from the dashboard layer. Before passing `schedule_df` to the logic functions, the dashboard applies:
1. `isCompleted == True` — exclude future/in-progress matchups
2. `isBye == False` — exclude BYE weeks (no opponent score to compare)
3. `isPlayoff` filter — based on the user's toggle state

This is consistent with the existing Team tab pattern (filter first, then pass to logic functions).

## Empty State

If filtering yields zero completed matchups, display `st.info("No completed matchups for the selected filter.")` and skip the metric cards and table entirely.

## Layout (top to bottom)

1. **Header:** "League Standings"
2. **Toggle:** Checkbox "Include playoff matchups" (default off)
3. **Metric cards:** 3x `st.metric` — league leader in PF, best All-Play record (by all-play win count), biggest point differential
4. **Standings table:** `st.dataframe` with all columns, `use_container_width=True`, `hide_index=True`. PF/PA/Diff formatted to 1 decimal place.

## File Changes

### `dashboard_logic.py`
- `compute_standings(schedule_df: pd.DataFrame) -> pd.DataFrame` — receives pre-filtered DataFrame, aggregates W-L-T, PF, PA, Diff per team
- `compute_all_play_record(schedule_df: pd.DataFrame) -> pd.DataFrame` — receives pre-filtered DataFrame, computes all-play W-L per team across matchup periods. Returns DataFrame with columns: `Team`, `AP_W`, `AP_L`, `AP_T`

The standings tab merges the two results on `Team` before display. Both functions rename `teamName` to `Team` internally for display-friendly output.

### `dashboard.py`

**What moves above tabs:**
- Configuration loading (`st.secrets`, config dict, `st.stop()` guards) — renders above tabs as app-level setup
- `ensure_data_files_exist()` call and the spinner wrapping it
- Data freshness row and refresh button
- `schedule_df = team_schedule_to_dataframe(schedule_payload)` and `schedule_generated_at` computation

**What stays inside `draft_tab`:**
- `process_data()` call and its error guards (only needed by Draft tab)
- All draft-specific rendering (scatter plot, value tables, waiver analysis)

**What stays inside `team_tab`:**
- Team-specific filters, metrics, charts, and schedule rendering (unchanged)

**New:**
- Tab unpacking: `draft_tab, standings_tab, team_tab = st.tabs([...])`
- `with standings_tab:` block implementing the layout above

### `tests/test_dashboard_logic.py`
- `compute_standings`: basic W-L-T aggregation with known data, empty DataFrame input
- `compute_all_play_record`: 3-team 2-period scenario verifiable by hand, empty DataFrame, tied scores between teams in a period

## No New Dependencies

All computation uses pandas operations on existing data. No new packages required.
