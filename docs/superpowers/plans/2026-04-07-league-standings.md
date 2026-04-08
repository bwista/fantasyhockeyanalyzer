# League Standings Tab Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Standings" tab (positioned second) showing league-wide W-L-T records, points for/against, and an all-play record computed from existing schedule data.

**Architecture:** Two new pure functions in `dashboard_logic.py` (`compute_standings`, `compute_all_play_record`) receive a pre-filtered `schedule_df` and return DataFrames. The dashboard layer handles filtering, merges the results, and renders a metric cards row + standings table. Data loading is hoisted above tab definitions so all three tabs share it.

**Tech Stack:** Python, pandas, Streamlit, pytest

**Spec:** `docs/superpowers/specs/2026-04-07-league-standings-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/dashboard/dashboard_logic.py` | Modify | Add `compute_standings()` and `compute_all_play_record()` |
| `src/dashboard/dashboard.py` | Modify | Hoist shared data loading, add standings tab |
| `tests/test_dashboard_logic.py` | Modify | Add tests for the two new functions |

---

### Task 1: Add `compute_standings` with tests

**Files:**
- Modify: `tests/test_dashboard_logic.py`
- Modify: `src/dashboard/dashboard_logic.py`

- [ ] **Step 1: Write failing tests for `compute_standings`**

Add to `tests/test_dashboard_logic.py`:

```python
from src.dashboard.dashboard_logic import (
    compute_standings, compute_duration_and_avg, process_data,
    get_data_freshness, get_scoring_categories, get_top_contributors,
)


def test_compute_standings_basic():
    """Aggregates W-L-T, PF, PA, Diff per team from completed matchups."""
    df = pd.DataFrame({
        'teamName': ['Alpha', 'Alpha', 'Beta', 'Beta'],
        'result': ['W', 'L', 'L', 'W'],
        'teamScore': [120.0, 90.0, 80.0, 110.0],
        'opponentScore': [80.0, 110.0, 120.0, 90.0],
    })
    result = compute_standings(df)
    alpha = result[result['Team'] == 'Alpha'].iloc[0]
    assert alpha['W'] == 1
    assert alpha['L'] == 1
    assert alpha['T'] == 0
    assert alpha['PF'] == pytest.approx(210.0)
    assert alpha['PA'] == pytest.approx(190.0)
    assert alpha['Diff'] == pytest.approx(20.0)


def test_compute_standings_with_ties():
    """Ties are counted correctly."""
    df = pd.DataFrame({
        'teamName': ['Alpha', 'Beta'],
        'result': ['T', 'T'],
        'teamScore': [100.0, 100.0],
        'opponentScore': [100.0, 100.0],
    })
    result = compute_standings(df)
    assert result.iloc[0]['T'] == 1


def test_compute_standings_sorted_by_wins_then_pf():
    """Teams sorted by W desc, then PF desc, then Diff desc."""
    df = pd.DataFrame({
        'teamName': ['Alpha', 'Alpha', 'Beta', 'Beta', 'Gamma', 'Gamma'],
        'result': ['W', 'W', 'W', 'L', 'W', 'L'],
        'teamScore': [100.0, 100.0, 210.0, 50.0, 120.0, 80.0],
        'opponentScore': [80.0, 80.0, 50.0, 150.0, 80.0, 120.0],
    })
    result = compute_standings(df)
    assert result.iloc[0]['Team'] == 'Alpha'  # 2W, 200 PF
    assert result.iloc[1]['Team'] == 'Beta'   # 1W, 260 PF (tiebreaker over Gamma)
    assert result.iloc[2]['Team'] == 'Gamma'  # 1W, 200 PF


def test_compute_standings_empty_df():
    """Empty input returns empty DataFrame with correct columns."""
    df = pd.DataFrame(columns=['teamName', 'result', 'teamScore', 'opponentScore'])
    result = compute_standings(df)
    assert result.empty
    assert 'Team' in result.columns
    assert 'W' in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_logic.py::test_compute_standings_basic tests/test_dashboard_logic.py::test_compute_standings_with_ties tests/test_dashboard_logic.py::test_compute_standings_sorted_by_wins_then_pf tests/test_dashboard_logic.py::test_compute_standings_empty_df -v`
Expected: FAIL with `ImportError` — `compute_standings` does not exist yet.

- [ ] **Step 3: Implement `compute_standings`**

Add to `src/dashboard/dashboard_logic.py`:

```python
def compute_standings(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate W-L-T, PF, PA, Diff per team from a pre-filtered schedule DataFrame.
    Expects columns: teamName, result, teamScore, opponentScore.
    Returns DataFrame sorted by W desc, PF desc, Diff desc.
    """
    if schedule_df.empty:
        return pd.DataFrame(columns=['Team', 'W', 'L', 'T', 'PF', 'PA', 'Diff'])

    grouped = schedule_df.groupby('teamName').agg(
        W=('result', lambda x: (x == 'W').sum()),
        L=('result', lambda x: (x == 'L').sum()),
        T=('result', lambda x: (x == 'T').sum()),
        PF=('teamScore', 'sum'),
        PA=('opponentScore', 'sum'),
    ).reset_index()

    grouped.rename(columns={'teamName': 'Team'}, inplace=True)
    grouped['Diff'] = grouped['PF'] - grouped['PA']
    grouped.sort_values(['W', 'PF', 'Diff'], ascending=[False, False, False], inplace=True)
    grouped.reset_index(drop=True, inplace=True)
    return grouped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_logic.py -k "compute_standings" -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_dashboard_logic.py src/dashboard/dashboard_logic.py
git commit -m "feat: add compute_standings function with tests"
```

---

### Task 2: Add `compute_all_play_record` with tests

**Files:**
- Modify: `tests/test_dashboard_logic.py`
- Modify: `src/dashboard/dashboard_logic.py`

- [ ] **Step 1: Write failing tests for `compute_all_play_record`**

Add to `tests/test_dashboard_logic.py`:

```python
Add `compute_all_play_record` to the existing import line (the one updated in Task 1).


def test_compute_all_play_basic():
    """3 teams, 1 period: highest scorer gets 2 wins, lowest gets 2 losses."""
    df = pd.DataFrame({
        'teamName': ['Alpha', 'Beta', 'Gamma'],
        'matchupPeriod': [1, 1, 1],
        'teamScore': [100.0, 80.0, 120.0],
    })
    result = compute_all_play_record(df)
    gamma = result[result['Team'] == 'Gamma'].iloc[0]
    assert gamma['AP_W'] == 2
    assert gamma['AP_L'] == 0
    beta = result[result['Team'] == 'Beta'].iloc[0]
    assert beta['AP_W'] == 0
    assert beta['AP_L'] == 2


def test_compute_all_play_multiple_periods():
    """All-play wins/losses aggregate across periods."""
    df = pd.DataFrame({
        'teamName': ['Alpha', 'Beta', 'Alpha', 'Beta'],
        'matchupPeriod': [1, 1, 2, 2],
        'teamScore': [100.0, 80.0, 70.0, 90.0],
    })
    result = compute_all_play_record(df)
    alpha = result[result['Team'] == 'Alpha'].iloc[0]
    assert alpha['AP_W'] == 1  # Won period 1, lost period 2
    assert alpha['AP_L'] == 1


def test_compute_all_play_tied_scores():
    """Tied scores count as all-play ties."""
    df = pd.DataFrame({
        'teamName': ['Alpha', 'Beta'],
        'matchupPeriod': [1, 1],
        'teamScore': [100.0, 100.0],
    })
    result = compute_all_play_record(df)
    assert result.iloc[0]['AP_T'] == 1
    assert result.iloc[0]['AP_W'] == 0
    assert result.iloc[0]['AP_L'] == 0


def test_compute_all_play_empty_df():
    """Empty input returns empty DataFrame with correct columns."""
    df = pd.DataFrame(columns=['teamName', 'matchupPeriod', 'teamScore'])
    result = compute_all_play_record(df)
    assert result.empty
    assert 'Team' in result.columns
    assert 'AP_W' in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_logic.py -k "compute_all_play" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `compute_all_play_record`**

Add to `src/dashboard/dashboard_logic.py`:

```python
def compute_all_play_record(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all-play W-L-T for each team across matchup periods.
    For each period, each team is compared against every other team's score.
    Expects columns: teamName, matchupPeriod, teamScore.
    Returns DataFrame with columns: Team, AP_W, AP_L, AP_T.
    """
    if schedule_df.empty:
        return pd.DataFrame(columns=['Team', 'AP_W', 'AP_L', 'AP_T'])

    records = []
    for period, group in schedule_df.groupby('matchupPeriod'):
        scores = group[['teamName', 'teamScore']].dropna()
        for _, row in scores.iterrows():
            team = row['teamName']
            score = row['teamScore']
            others = scores[scores['teamName'] != team]['teamScore']
            records.append({
                'Team': team,
                'AP_W': int((others < score).sum()),
                'AP_L': int((others > score).sum()),
                'AP_T': int((others == score).sum()),
            })

    if not records:
        return pd.DataFrame(columns=['Team', 'AP_W', 'AP_L', 'AP_T'])

    result = pd.DataFrame(records).groupby('Team', as_index=False).agg(
        AP_W=('AP_W', 'sum'),
        AP_L=('AP_L', 'sum'),
        AP_T=('AP_T', 'sum'),
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_logic.py -k "compute_all_play" -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_dashboard_logic.py src/dashboard/dashboard_logic.py
git commit -m "feat: add compute_all_play_record function with tests"
```

---

### Task 3: Hoist shared data loading in `dashboard.py`

**Files:**
- Modify: `src/dashboard/dashboard.py`

This task restructures `dashboard.py` so that data loading happens above the tab definitions. No logic changes — just moving existing code.

- [ ] **Step 1: Move config loading above tabs**

Move the configuration block (currently lines 49-78 inside `with draft_tab:`) to the top of the file, after `st.header(...)` and before the tab definitions. This includes:
- `config_placeholder` + secrets loading + `st.stop()` guards
- `config_placeholder.empty()`
- The `if not all([...])` guard

- [ ] **Step 2: Move data loading above tabs**

Move these blocks (currently inside `with draft_tab:`) to above the tab definitions:
- `ensure_data_files_exist()` call with its spinner
- Freshness row (`get_data_freshness`, caption, refresh button)
- `schedule_df = team_schedule_to_dataframe(schedule_payload)`
- `schedule_generated_at` computation

- [ ] **Step 3: Update tab definitions**

Change:
```python
draft_tab, team_tab = st.tabs(["📊 Draft", "🏒 Team"])
```
To:
```python
draft_tab, standings_tab, team_tab = st.tabs(["📊 Draft", "🏆 Standings", "🏒 Team"])
```

- [ ] **Step 4: Verify nothing broke**

Run: `pytest tests/ -v`
Expected: All existing tests PASS.

Run: `streamlit run src/dashboard/dashboard.py` (manual check — Draft and Team tabs still work)

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/dashboard.py
git commit -m "refactor: hoist shared data loading above tab definitions"
```

---

### Task 4: Build the standings tab UI

**Files:**
- Modify: `src/dashboard/dashboard.py`
- Modify: `src/dashboard/dashboard_logic.py` (update imports in `__init__` / function exports)

- [ ] **Step 1: Add imports in `dashboard.py`**

Add `compute_standings` and `compute_all_play_record` to the import block:

```python
from src.dashboard.dashboard_logic import (
    ensure_data_files_exist, process_data, plot_draft_value, team_schedule_to_dataframe,
    plot_matchup_scores_by_period, compute_duration_and_avg, get_acquiring_teams, get_data_freshness,
    get_scoring_categories, get_top_contributors,
    compute_standings, compute_all_play_record,
)
```

- [ ] **Step 2: Implement the standings tab block**

Add `with standings_tab:` block between `draft_tab` and `team_tab`:

```python
with standings_tab:
    st.header("League Standings")

    if schedule_df is not None and not schedule_df.empty:
        include_playoffs_standings = st.checkbox(
            "Include playoff matchups",
            value=False,
            help="Include playoff matchups in standings calculations.",
            key="standings_include_playoffs",
        )

        # Pre-filter
        standings_source = schedule_df[
            (schedule_df['isCompleted']) & (~schedule_df['isBye'])
        ].copy()
        if not include_playoffs_standings:
            standings_source = standings_source[~standings_source['isPlayoff']]

        if standings_source.empty:
            st.info("No completed matchups for the selected filter.")
        else:
            standings_df = compute_standings(standings_source)
            all_play_df = compute_all_play_record(standings_source)
            merged = standings_df.merge(all_play_df, on='Team', how='left')
            merged['AP_W'] = merged['AP_W'].fillna(0).astype(int)
            merged['AP_L'] = merged['AP_L'].fillna(0).astype(int)
            merged['AP_T'] = merged['AP_T'].fillna(0).astype(int)
            merged['All-Play'] = merged.apply(
                lambda r: f"{r['AP_W']}-{r['AP_L']}" + (f"-{r['AP_T']}" if r['AP_T'] > 0 else ""),
                axis=1,
            )

            # Metric cards
            best_pf = merged.loc[merged['PF'].idxmax()]
            best_ap = merged.loc[merged['AP_W'].idxmax()]
            best_diff = merged.loc[merged['Diff'].idxmax()]

            m1, m2, m3 = st.columns(3)
            m1.metric("Most Points For", f"{best_pf['PF']:.1f}", delta=best_pf['Team'])
            m2.metric("Best All-Play", f"{best_ap['AP_W']}-{best_ap['AP_L']}", delta=best_ap['Team'])
            m3.metric("Best Pt Diff", f"{best_diff['Diff']:+.1f}", delta=best_diff['Team'])

            # Standings table
            display_cols = ['Team', 'W', 'L', 'T', 'PF', 'PA', 'Diff', 'All-Play']
            st.dataframe(
                merged[display_cols].style.format({'PF': '{:.1f}', 'PA': '{:.1f}', 'Diff': '{:+.1f}'}),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Schedule data is not available.")
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Manual smoke test**

Run: `streamlit run src/dashboard/dashboard.py`
Verify:
- Standings tab appears between Draft and Team
- Table shows all teams with W-L-T, PF, PA, Diff, All-Play
- Playoff toggle works
- Metric cards show correct leaders
- Draft and Team tabs still work as before

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/dashboard.py
git commit -m "feat: add league standings tab with all-play record"
```

---

### Task 5: Run full test suite and final verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 2: Run linting (if configured)**

Run: `python -m py_compile src/dashboard/dashboard.py && python -m py_compile src/dashboard/dashboard_logic.py`
Expected: No errors.

- [ ] **Step 3: Final commit (if any cleanup needed)**

Only if previous steps revealed issues to fix.
