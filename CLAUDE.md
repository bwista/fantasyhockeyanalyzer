# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the dashboard
streamlit run src/dashboard/dashboard.py

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test file
pytest tests/test_calculate_scoring_system.py

# Run tests with coverage
pytest --cov=src --cov-report=html
```

## Configuration

Local development requires `user_config.json` in the project root:
```json
{
  "LEAGUE_ID": 12345678,
  "YEAR": 2025,
  "SWID": "{YOUR_SWID_COOKIE_VALUE}",
  "ESPN_S2": "YOUR_ESPN_S2_COOKIE_VALUE"
}
```

For Streamlit Cloud deployment, these values are stored in `.streamlit/secrets.toml` instead.

## Architecture

The app is a Streamlit dashboard for ESPN fantasy hockey analysis. Data flows as:

**ESPN API → JSON cache → processing → dashboard**

### Data Layer (`src/data_processing/`)
- `parse_draft_results.py` — fetches draft picks via `espn_api` library
- `fetch_box_score_stats.py` — fetches weekly player stats (box scores, week 1 to current)
- `fetch_team_info.py` — maps team names to abbreviations
- `fetch_team_schedule.py` — fetches league schedules and matchup data

All results are persisted to `src/data/*.json`. The dashboard auto-fetches if these files are missing.

### Analysis (`src/analysis/`)
- `calculate_scoring_system.py` — derives per-stat scoring weights from box score data

### Dashboard (`src/dashboard/`)
- `dashboard.py` — main Streamlit app with two tabs:
  - **Draft tab**: scatter plot of pick # vs. fantasy points rank, best/worst value picks, waiver acquisitions, team-specific analysis
  - **Team tab**: schedules, matchup history, performance over season
- `dashboard_logic.py` — business logic: data loading, stat aggregation, plotting, schedule parsing

### Key Calculations
- **ValueScore** = DraftRank (overall pick #) − PointsRank (performance rank). Positive = outperforming draft position.
- **PointsRank** = rank by TotalPoints across all players
- **Acquisition type**: player is 'drafted' if their first recorded team matches the drafting team, otherwise 'waiver'
- OLS trendline via `statsmodels` on pick # vs. rank scatter
