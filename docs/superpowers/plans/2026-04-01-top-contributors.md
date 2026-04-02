# Top Contributors Feature — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show each team's top point contributors on the Team tab, with an MVP highlight card and a top-5 leaderboard, filterable by matchup period.

**Architecture:** Add two pure functions to `dashboard_logic.py` — one to derive scoring categories from the data, one to compute top contributors. Wire them into the existing team tab after "Recent Results", reusing the team/period filters already in place.

**Tech Stack:** Python, pandas, Streamlit

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/dashboard/dashboard_logic.py` | Add `get_scoring_categories()` and `get_top_contributors()` |
| Modify | `src/dashboard/dashboard.py` | Import new functions, add MVP card + leaderboard table to team tab |
| Modify | `tests/test_dashboard_logic.py` | Tests for both new logic functions |

---

### Task 1: Add `get_scoring_categories()` to `dashboard_logic.py`

**Files:**
- Modify: `src/dashboard/dashboard_logic.py` (append after `get_acquiring_teams`)
- Test: `tests/test_dashboard_logic.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_dashboard_logic.py`:

```python
from src.dashboard.dashboard_logic import get_scoring_categories


def test_get_scoring_categories_returns_nonzero_keys():
    """Only stat categories with non-zero fantasy point values are returned."""
    stats_df = pd.DataFrame({
        'points_breakdown': [
            {'G': 2.0, 'A': 4.0, 'PPP': 0.0, 'HIT': 0.5},
            {'G': 0.0, 'A': 2.0, 'PPP': 1.0, 'HIT': 0.0},
        ]
    })
    result = get_scoring_categories(stats_df)
    assert result == ['A', 'G', 'HIT', 'PPP']


def test_get_scoring_categories_empty_df():
    """Empty DataFrame returns empty list."""
    stats_df = pd.DataFrame({'points_breakdown': []})
    result = get_scoring_categories(stats_df)
    assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_logic.py::test_get_scoring_categories_returns_nonzero_keys tests/test_dashboard_logic.py::test_get_scoring_categories_empty_df -v`
Expected: FAIL — `ImportError: cannot import name 'get_scoring_categories'`

- [ ] **Step 3: Implement `get_scoring_categories`**

Append to end of `src/dashboard/dashboard_logic.py` (after `get_acquiring_teams`):

```python
def get_scoring_categories(stats_df: pd.DataFrame) -> list:
    """Return sorted list of stat keys that have non-zero fantasy point values across the dataset."""
    if stats_df is None or stats_df.empty or 'points_breakdown' not in stats_df.columns:
        return []
    nonzero_keys = set()
    for pb in stats_df['points_breakdown'].dropna():
        if isinstance(pb, dict):
            for key, val in pb.items():
                if val != 0:
                    nonzero_keys.add(key)
    return sorted(nonzero_keys)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_logic.py::test_get_scoring_categories_returns_nonzero_keys tests/test_dashboard_logic.py::test_get_scoring_categories_empty_df -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/dashboard_logic.py tests/test_dashboard_logic.py
git commit -m "feat: add get_scoring_categories to derive scoring stats from data"
```

---

### Task 2: Add `get_top_contributors()` to `dashboard_logic.py`

**Files:**
- Modify: `src/dashboard/dashboard_logic.py` (append after `get_scoring_categories`)
- Test: `tests/test_dashboard_logic.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_dashboard_logic.py`:

```python
from src.dashboard.dashboard_logic import get_top_contributors


def test_get_top_contributors_basic():
    """Returns players sorted by total_points descending with aggregated raw stats."""
    stats_df = pd.DataFrame({
        'name': ['Alice', 'Alice', 'Bob', 'Bob'],
        'position': ['Center', 'Center', 'Defense', 'Defense'],
        'team_abbrev': ['ODB', 'ODB', 'ODB', 'ODB'],
        'week': [1, 2, 1, 2],
        'total_points': [10.0, 15.0, 20.0, 5.0],
        'points_breakdown': [
            {'G': 2.0, 'A': 4.0},
            {'G': 4.0, 'A': 2.0},
            {'G': 6.0, 'A': 0.0},
            {'G': 0.0, 'A': 2.0},
        ],
        'stats': [
            {'G': 1.0, 'A': 2.0},
            {'G': 2.0, 'A': 1.0},
            {'G': 3.0, 'A': 0.0},
            {'G': 0.0, 'A': 1.0},
        ],
    })
    scoring_cats = ['A', 'G']
    result = get_top_contributors(stats_df, 'ODB', 1, 2, scoring_cats, top_n=5)
    assert len(result) == 2
    assert set(result['Player'].tolist()) == {'Alice', 'Bob'}
    assert result['TotalPoints'].sum() == pytest.approx(50.0)
    assert 'G' in result.columns
    assert 'A' in result.columns


def test_get_top_contributors_filters_by_team():
    """Only players from the selected team are included."""
    stats_df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'position': ['Center', 'Defense'],
        'team_abbrev': ['ODB', 'JFA'],
        'week': [1, 1],
        'total_points': [10.0, 20.0],
        'points_breakdown': [{'G': 2.0}, {'G': 4.0}],
        'stats': [{'G': 1.0}, {'G': 2.0}],
    })
    result = get_top_contributors(stats_df, 'ODB', 1, 1, ['G'], top_n=5)
    assert len(result) == 1
    assert result.iloc[0]['Player'] == 'Alice'


def test_get_top_contributors_filters_by_week_range():
    """Only weeks within range are included."""
    stats_df = pd.DataFrame({
        'name': ['Alice', 'Alice', 'Alice'],
        'position': ['Center', 'Center', 'Center'],
        'team_abbrev': ['ODB', 'ODB', 'ODB'],
        'week': [1, 2, 3],
        'total_points': [10.0, 20.0, 30.0],
        'points_breakdown': [{'G': 2.0}, {'G': 4.0}, {'G': 6.0}],
        'stats': [{'G': 1.0}, {'G': 2.0}, {'G': 3.0}],
    })
    result = get_top_contributors(stats_df, 'ODB', 1, 2, ['G'], top_n=5)
    assert result.iloc[0]['TotalPoints'] == pytest.approx(30.0)
    assert result.iloc[0]['G'] == pytest.approx(3.0)


def test_get_top_contributors_respects_top_n():
    """Only top N players returned."""
    rows = []
    for i in range(10):
        rows.append({
            'name': f'Player{i}', 'position': 'Center', 'team_abbrev': 'ODB',
            'week': 1, 'total_points': float(i), 'points_breakdown': {'G': 0.0},
            'stats': {'G': 0.0},
        })
    stats_df = pd.DataFrame(rows)
    result = get_top_contributors(stats_df, 'ODB', 1, 1, ['G'], top_n=3)
    assert len(result) == 3
    assert result.iloc[0]['TotalPoints'] == pytest.approx(9.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_logic.py -k "test_get_top_contributors" -v`
Expected: FAIL — `ImportError: cannot import name 'get_top_contributors'`

- [ ] **Step 3: Implement `get_top_contributors`**

Add to `src/dashboard/dashboard_logic.py` after `get_scoring_categories`:

```python
def get_top_contributors(
    stats_df: pd.DataFrame,
    team_abbrev: str,
    start_week: int,
    end_week: int,
    scoring_categories: list,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return top N contributors for a team within a week range, with aggregated raw stats."""
    if stats_df is None or stats_df.empty:
        return pd.DataFrame()

    filtered = stats_df[
        (stats_df['team_abbrev'] == team_abbrev)
        & (stats_df['week'] >= start_week)
        & (stats_df['week'] <= end_week)
    ].copy()

    if filtered.empty:
        return pd.DataFrame()

    # Expand raw stats dict into columns for each scoring category
    for cat in scoring_categories:
        filtered[cat] = filtered['stats'].apply(lambda s: s.get(cat, 0.0) if isinstance(s, dict) else 0.0)

    # Group by player, sum points and raw stat counts
    agg_dict = {'total_points': 'sum'}
    for cat in scoring_categories:
        agg_dict[cat] = 'sum'

    grouped = (
        filtered.groupby(['name', 'position'])
        .agg(agg_dict)
        .reset_index()
        .sort_values('total_points', ascending=False)
        .head(top_n)
    )

    # Map position to short form
    position_map = {
        'Left Wing': 'F', 'Right Wing': 'F', 'Center': 'F',
        'Defense': 'D', 'Goalie': 'G',
    }
    grouped['position'] = grouped['position'].replace(position_map)

    grouped.rename(columns={'name': 'Player', 'position': 'Pos', 'total_points': 'TotalPoints'}, inplace=True)

    return grouped.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_logic.py -k "test_get_top_contributors" -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/dashboard_logic.py tests/test_dashboard_logic.py
git commit -m "feat: add get_top_contributors with raw stat aggregation"
```

---

### Task 3: Wire top contributors into team tab UI

**Files:**
- Modify: `src/dashboard/dashboard.py:21` (update import)
- Modify: `src/dashboard/dashboard.py:335-346` (insert section after "Recent Results", before "Full Schedule")

- [ ] **Step 1: Update import in `dashboard.py`**

At line 21, add `get_top_contributors` and `get_scoring_categories` to the import:

```python
from src.dashboard.dashboard_logic import (
    ensure_data_files_exist, process_data, plot_draft_value, team_schedule_to_dataframe,
    plot_matchup_scores_by_period, compute_duration_and_avg, get_acquiring_teams, get_data_freshness,
    get_scoring_categories, get_top_contributors
)
```

- [ ] **Step 2: Add top contributors section to team tab**

Insert after the "Recent Results" section (after line 342) and before the "Full Schedule" expander (line 344):

```python
                # --- Top Contributors ---
                st.subheader("Top Contributors")
                if stats_df is not None and team_map is not None:
                    selected_abbrev = team_map.get(selected_team_name)

                    if selected_abbrev:
                        scoring_cats = get_scoring_categories(stats_df)
                        contributors_df = get_top_contributors(
                            stats_df, selected_abbrev, start_period, end_period,
                            scoring_cats, top_n=5
                        )

                        if not contributors_df.empty:
                            # MVP highlight card
                            mvp = contributors_df.iloc[0]
                            period_label = (
                                f"Week {start_period}" if start_period == end_period
                                else f"Weeks {start_period}-{end_period}"
                            )
                            stat_line = "  ".join(
                                f"{mvp[cat]:.0f}{cat}" if mvp[cat] == int(mvp[cat])
                                else f"{mvp[cat]:.1f}{cat}"
                                for cat in scoring_cats if cat in mvp.index
                            )
                            st.markdown(
                                f"**Top Contributor ({period_label})**\n\n"
                                f"### {mvp['Player']} ({mvp['Pos']}) — {mvp['TotalPoints']:.1f} pts\n\n"
                                f"`{stat_line}`"
                            )

                            # Leaderboard table
                            display_cols = ['Player', 'Pos', 'TotalPoints'] + [
                                c for c in scoring_cats if c in contributors_df.columns
                            ]
                            st.dataframe(
                                contributors_df[display_cols],
                                hide_index=True,
                                width='stretch',
                            )
                        else:
                            st.info("No player stats available for the selected filters.")
                    else:
                        st.info("Could not map team name to abbreviation.")
                else:
                    st.info("Player stats data not available for top contributors.")
```

- [ ] **Step 3: Manually test in the browser**

Run: `streamlit run src/dashboard/dashboard.py`
- Navigate to Team tab
- Select a team, verify MVP card and leaderboard appear
- Adjust the matchup period slider to a single week, verify it updates
- Expand to full season range, verify season totals

- [ ] **Step 4: Commit**

```bash
git add src/dashboard/dashboard.py
git commit -m "feat: add top contributors section to team tab with MVP card and leaderboard"
```
