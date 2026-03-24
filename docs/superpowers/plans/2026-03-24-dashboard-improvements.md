# Dashboard Improvements Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers-extended-cc:subagent-driven-development (if subagents available) or superpowers-extended-cc:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix bugs, remove dead code, and improve UX consistency in the Fantasy Hockey Analyzer dashboard.

**Architecture:** All business logic lives in `dashboard_logic.py`; `dashboard.py` is pure UI wiring. Improvements are grouped into: bug fixes (off-by-one, exception handling), code cleanup (dead duplicate functions), and UX polish (consistent team selectors, readable column names, non-blocking UI, data freshness controls).

**Tech Stack:** Python 3.8+, Streamlit, Pandas, Plotly, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/dashboard/dashboard.py` | Modify | Remove dead functions; fix Duration calc; use full team names in waiver selector; rename display columns; remove `time.sleep`; add refresh button + timestamp |
| `src/dashboard/dashboard_logic.py` | Modify | Add `compute_duration_and_avg`, `get_acquiring_teams`, `get_data_freshness` helpers; narrow bare exception in `process_data`; move constants inside `ensure_data_files_exist` |
| `tests/test_dashboard_logic.py` | Create | Unit tests for `compute_duration_and_avg` and the narrowed exception path |
| `tests/test_dashboard_display.py` | Create | Tests for column rename logic and team name consistency helpers |

---

## Task 1: Fix AvgPointsPerWeek Off-by-One Bug

**Context:** Duration is computed as `LastWeek - FirstWeek`, which is off by one. A player on a team weeks 1–10 has duration 9, not 10. This inflates every average.

**Files:**
- Create: `tests/test_dashboard_logic.py`
- Modify: `src/dashboard/dashboard_logic.py`
- Modify: `src/dashboard/dashboard.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_dashboard_logic.py`:

```python
import pandas as pd
import pytest
from src.dashboard.dashboard_logic import compute_duration_and_avg


def test_duration_includes_both_endpoints():
    """Player on team weeks 1-10 has 10 weeks, not 9."""
    df = pd.DataFrame({
        'Player': ['A'],
        'TeamPoints': [100.0],
        'FirstWeek': [1],
        'LastWeek': [10],
    })
    result = compute_duration_and_avg(df)
    assert result.loc[0, 'Duration'] == 10
    assert result.loc[0, 'AvgPointsPerWeek'] == pytest.approx(10.0)


def test_duration_single_week():
    """Player on team for only 1 week: duration=1, avg=all their points."""
    df = pd.DataFrame({
        'Player': ['B'],
        'TeamPoints': [50.0],
        'FirstWeek': [5],
        'LastWeek': [5],
    })
    result = compute_duration_and_avg(df)
    assert result.loc[0, 'Duration'] == 1
    assert result.loc[0, 'AvgPointsPerWeek'] == pytest.approx(50.0)


def test_avg_points_zero_duration_guard():
    """If FirstWeek == LastWeek + 1 somehow (data error), do not divide by zero."""
    df = pd.DataFrame({
        'Player': ['C'],
        'TeamPoints': [80.0],
        'FirstWeek': [3],
        'LastWeek': [2],  # bad data
    })
    result = compute_duration_and_avg(df)
    assert result.loc[0, 'AvgPointsPerWeek'] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_dashboard_logic.py -v
```

Expected: `ImportError` or `AttributeError` — `compute_duration_and_avg` does not exist yet.

- [ ] **Step 3: Implement `compute_duration_and_avg` in `dashboard_logic.py`**

Add this function near the top of the `# --- Plotting/Display Functions ---` section (around line 416):

```python
def compute_duration_and_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Duration (inclusive week count) and AvgPointsPerWeek columns.
    Duration = LastWeek - FirstWeek + 1  (inclusive on both ends).
    Returns AvgPointsPerWeek = 0.0 for any row where Duration <= 0 (bad data guard).
    """
    df = df.copy()
    df['Duration'] = df['LastWeek'] - df['FirstWeek'] + 1
    df['AvgPointsPerWeek'] = df.apply(
        lambda r: r['TeamPoints'] / r['Duration'] if r['Duration'] > 0 else 0.0,
        axis=1
    )
    return df
```

**Note on bad-data test:** `FirstWeek=3, LastWeek=2` yields `Duration = 0`, so `AvgPointsPerWeek` is `0.0` via the lambda guard — not clipped to `1`. The test `test_avg_points_zero_duration_guard` asserts `0.0` which is consistent with this implementation.

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_dashboard_logic.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Replace the two inline Duration/AvgPointsPerWeek blocks in `dashboard.py`**

In `dashboard.py`, find the two occurrences of this pattern (lines ~325 and ~347):

```python
unique_waiver_players['Duration'] = unique_waiver_players['LastWeek'] - unique_waiver_players['FirstWeek']
unique_waiver_players['AvgPointsPerWeek'] = unique_waiver_players['TeamPoints'] / unique_waiver_players['Duration']
```

Replace each with:

```python
from src.dashboard.dashboard_logic import compute_duration_and_avg
# (add import at top of file alongside existing dashboard_logic imports)
unique_waiver_players = compute_duration_and_avg(unique_waiver_players)
```

And for the second occurrence:

```python
team_acquisitions_df = compute_duration_and_avg(team_acquisitions_df)
```

Also add `compute_duration_and_avg` to the import line at the top of `dashboard.py`:

```python
from src.dashboard.dashboard_logic import (
    ensure_data_files_exist, process_data, plot_draft_value, team_schedule_to_dataframe,
    plot_matchup_scores_by_period, compute_duration_and_avg
)
```

- [ ] **Step 6: Run full test suite**

```
pytest -v
```

Expected: all existing tests + new tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_dashboard_logic.py src/dashboard/dashboard_logic.py src/dashboard/dashboard.py
git commit -m "fix: correct AvgPointsPerWeek off-by-one (Duration now inclusive)"
```

---

## Task 2: Remove Dead Duplicate Functions from `dashboard.py`

**Context:** `dashboard.py` defines `load_data` (lines 60–110), `calculate_value` (lines 112–152), and `determine_acquisition_type` (lines 155–180). None of these are called — `process_data` in `dashboard_logic.py` uses its own internal versions. They are stale dead code.

**Files:**
- Modify: `src/dashboard/dashboard.py`

- [ ] **Step 1: Verify the functions are truly unused**

```bash
grep -n "load_data\|calculate_value\|determine_acquisition_type" src/dashboard/dashboard.py
```

Expected output: only the `def` lines appear, no call sites outside the function bodies.

- [ ] **Step 2: Delete the three dead functions**

Remove these blocks from `dashboard.py`:
- The `load_data` function (lines ~60–110)
- The `calculate_value` function (lines ~112–152)
- The `determine_acquisition_type` function (lines ~155–180)

**Do NOT remove `import time` in this task.** It is removed in Task 5 Step 3, after `show_temporary_message` (which uses it) is deleted. Removing `import time` here would break the app until Task 5 runs.

After deletion, the file should jump from the `# --- Configuration ---` block directly to `# --- Helper Functions ---` with only `show_temporary_message` remaining (until Task 5 removes that too).

- [ ] **Step 3: Run the app to confirm it still loads**

```bash
streamlit run src/dashboard/dashboard.py
```

Expected: dashboard loads without `NameError` or `ImportError`.

- [ ] **Step 4: Run full test suite**

```
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/dashboard.py
git commit -m "chore: remove dead duplicate functions from dashboard.py"
```

---

## Task 3: Narrow Bare Exception in `process_data`

**Context:** `process_data` in `dashboard_logic.py` wraps everything in `except Exception as e` and returns `(None, None)`, discarding the traceback. Debugging silently-failing data processing is painful.

**Files:**
- Modify: `src/dashboard/dashboard_logic.py`
- Modify: `tests/test_dashboard_logic.py`

- [ ] **Step 1: Write a test that verifies errors surface**

Add to `tests/test_dashboard_logic.py`:

```python
import pandas as pd
from src.dashboard.dashboard_logic import process_data


def test_process_data_raises_on_bad_stats_schema():
    """process_data should raise ValueError if stats_df is missing required columns."""
    draft_df = pd.DataFrame({
        'Player': ['A'],
        'DraftPick': [1],
        'Team': ['TOR'],
    })
    bad_stats_df = pd.DataFrame({'wrong_col': [1]})  # missing name, team_abbrev, etc.

    with pytest.raises((KeyError, ValueError)):
        process_data(draft_df, bad_stats_df, None)  # positional None for team_map
```

- [ ] **Step 2: Run to verify it fails (returns None instead of raising)**

```
pytest tests/test_dashboard_logic.py::test_process_data_raises_on_bad_stats_schema -v
```

Expected: FAIL — currently `process_data` catches the error and returns `(None, None)`.

- [ ] **Step 3: Replace the bare except in `process_data`**

In `dashboard_logic.py`, change the end of `process_data` from:

```python
    except Exception as e:
        if st: st.error(f"Error during data processing: {e}")
        return None, None
```

To:

```python
    except (KeyError, TypeError, AttributeError) as e:
        if st:
            st.error(f"Error during data processing: {e}")
        raise
```

This still shows the Streamlit error message when `st` is provided, but re-raises so callers (and tests) see the real exception.

Also update the call site in `dashboard.py`. The existing code at line ~258 is:

```python
        final_df, value_df = process_data(draft_df, stats_df, team_map, st=st)
```

Replace it with:

```python
        try:
            final_df, value_df = process_data(draft_df, stats_df, team_map, st=st)
        except Exception as e:
            st.error(f"Data processing failed: {e}")
            st.stop()
```

This is inside the `if draft_df is not None and stats_df is not None:` block — preserve that outer condition.

- [ ] **Step 4: Run test to verify it now passes**

```
pytest tests/test_dashboard_logic.py -v
```

Expected: all tests PASS including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/dashboard_logic.py src/dashboard/dashboard.py tests/test_dashboard_logic.py
git commit -m "fix: re-raise exceptions from process_data instead of silently returning None"
```

---

## Task 4: Fix Waiver Team Selector to Use Full Team Names

**Context:** The "Top Acquisitions by Acquiring Team" selectbox uses `team_abbrev` (e.g. "EDM") while every other team selector in the dashboard uses full names (e.g. "Edmonton Oilers"). This inconsistency forces users to mentally translate abbreviations. Extract the selector logic into a testable helper.

**Files:**
- Modify: `src/dashboard/dashboard_logic.py`
- Modify: `src/dashboard/dashboard.py`
- Create: `tests/test_dashboard_display.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_dashboard_display.py`:

```python
import pandas as pd
import pytest
from src.dashboard.dashboard_logic import get_acquiring_teams


def test_get_acquiring_teams_returns_full_names():
    """get_acquiring_teams should return sorted full team names, not abbreviations."""
    waiver_records_df = pd.DataFrame({
        'Player': ['A', 'B', 'C'],
        'team_abbrev': ['EDM', 'TOR', 'EDM'],
        'PickupTeamName': ['Edmonton Oilers', 'Toronto Maple Leafs', 'Edmonton Oilers'],
        'TeamPoints': [100, 80, 60],
    })
    result = get_acquiring_teams(waiver_records_df)
    assert result == ['Edmonton Oilers', 'Toronto Maple Leafs']
    assert 'EDM' not in result
    assert 'TOR' not in result


def test_get_acquiring_teams_drops_nulls():
    """Teams with missing PickupTeamName should be excluded."""
    waiver_records_df = pd.DataFrame({
        'PickupTeamName': ['Team A', None, 'Team B'],
        'TeamPoints': [10, 20, 30],
    })
    result = get_acquiring_teams(waiver_records_df)
    assert result == ['Team A', 'Team B']
```

- [ ] **Step 2: Run to verify it fails**

```
pytest tests/test_dashboard_display.py -v
```

Expected: `ImportError` — `get_acquiring_teams` does not exist yet.

- [ ] **Step 3: Add `get_acquiring_teams` to `dashboard_logic.py`**

Add this function near the bottom of `dashboard_logic.py`, before the final comment line:

```python
def get_acquiring_teams(waiver_df: pd.DataFrame) -> list:
    """Returns sorted list of unique full team names from PickupTeamName column."""
    return sorted(waiver_df['PickupTeamName'].dropna().unique().tolist())
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_dashboard_display.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Update the selectbox and filter in `dashboard.py`**

Add `get_acquiring_teams` to the import from `dashboard_logic` at the top of `dashboard.py`:

```python
from src.dashboard.dashboard_logic import (
    ensure_data_files_exist, process_data, plot_draft_value, team_schedule_to_dataframe,
    plot_matchup_scores_by_period, compute_duration_and_avg, get_acquiring_teams
)
```

(`get_data_freshness` is added to this import line in Task 7 Step 4 — do not add it here.)

Find the "Top Acquisitions by Acquiring Team" section (around line 340). Change:

```python
acquiring_teams = sorted(waiver_records_df['team_abbrev'].dropna().unique())
if acquiring_teams:
    selected_acq_team = st.selectbox('Select Acquiring Team:', acquiring_teams)
    if selected_acq_team:
        team_acquisitions_df = waiver_records_df[waiver_records_df['team_abbrev'] == selected_acq_team].copy()
```

To:

```python
acquiring_teams = get_acquiring_teams(waiver_records_df)
if acquiring_teams:
    selected_acq_team = st.selectbox('Select Acquiring Team:', acquiring_teams)
    if selected_acq_team:
        team_acquisitions_df = waiver_records_df[waiver_records_df['PickupTeamName'] == selected_acq_team].copy()
```

- [ ] **Step 6: Run full test suite**

```
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/dashboard/dashboard.py src/dashboard/dashboard_logic.py tests/test_dashboard_display.py
git commit -m "fix: use full team names in waiver acquiring team selector"
```

---

## Task 5: Remove Blocking `time.sleep` from Data Loading

**Context:** `show_temporary_message` calls `time.sleep(duration)` which blocks Streamlit's rendering thread. The 0.25s delay is too short to read anyway. Remove the helper entirely and replace with a `st.spinner` for the data loading section.

**Files:**
- Modify: `src/dashboard/dashboard.py`

- [ ] **Step 1: Delete `show_temporary_message` and its calls**

Remove the function definition (lines ~48–58):

```python
def show_temporary_message(message_type, content, duration):
    ...
```

Then search for every call to `show_temporary_message` in `dashboard_logic.py` and `dashboard.py` and remove them:

```bash
grep -n "show_temporary_message" src/dashboard/dashboard.py src/dashboard/dashboard_logic.py
```

Delete each call line. These were `st.success` messages that flash for 0.25s on load — they provide no value to the user.

- [ ] **Step 2: Wrap the data loading block in a spinner**

In `dashboard.py`, the `data_loading_placeholder` block (lines ~225–248) currently shows status messages that disappear. Replace the container with a spinner:

```python
with st.spinner("Loading data..."):
    draft_df, stats_df, team_map, schedule_payload = ensure_data_files_exist(
        config,
        DRAFT_RESULTS_FILE,
        PLAYER_STATS_FILE,
        TEAM_MAPPING_FILE,
        TEAM_SCHEDULE_FILE,
        parse_draft_results,
        fetch_box_score_stats,
        fetch_and_save_team_info,
        fetch_and_save_team_schedule,
        START_WEEK,
        END_WEEK,
        RATE_LIMIT_DELAY,
        TEAM_INFO_OUTPUT_DIR,
        TEAM_INFO_OUTPUT_FILE,
        st=st
    )
```

Remove the `data_loading_placeholder = st.empty()` and `data_loading_placeholder.empty()` wrapper lines.

- [ ] **Step 3: Remove `import time` from `dashboard.py`**

Since `show_temporary_message` was the only user of `time`, remove:
```python
import time # Added for temporary messages
```

- [ ] **Step 4: Run the app to verify no regression**

```bash
streamlit run src/dashboard/dashboard.py
```

Expected: dashboard loads cleanly with a spinner instead of flashing messages.

- [ ] **Step 5: Run full test suite**

```
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/dashboard.py src/dashboard/dashboard_logic.py
git commit -m "fix: remove blocking time.sleep from data loading, use spinner instead"
```

---

## Task 6: Rename Internal Columns in "Full Processed Data" Table

**Context:** The "Full Processed Data" expander shows raw internal column names like `team_abbrev`, `acquisition_type`, `DraftingTeamAbbrev` that are meaningless to a fantasy hockey user.

**Files:**
- Modify: `src/dashboard/dashboard.py`

- [ ] **Step 1: Write a test for the rename mapping**

Add to `tests/test_dashboard_display.py`:

```python
def test_full_data_display_renames_internal_columns():
    """Displayed column names should be human-readable, not internal identifiers."""
    internal_cols = ['team_abbrev', 'acquisition_type', 'DraftingTeamAbbrev']
    display_rename_map = {
        'team_abbrev': 'Current Team',
        'acquisition_type': 'How Acquired',
        'DraftingTeamAbbrev': 'Drafted By (Abbrev)',
    }
    for col in internal_cols:
        assert col in display_rename_map, f"{col} should have a display name"
        assert display_rename_map[col] != col, f"{col} rename should differ from original"
```

- [ ] **Step 2: Run to verify test passes**

```
pytest tests/test_dashboard_display.py -v
```

Expected: PASS (pure data test).

- [ ] **Step 3: Apply rename in the Full Processed Data expander**

In `dashboard.py`, inside the `with st.expander("Show Full Data Table"):` block (around line 362), after building `all_cols_ordered + remaining_cols`, add a rename step before `st.dataframe`:

```python
with st.expander("Show Full Data Table"):
    all_cols_ordered = [
        'Player', 'team_abbrev', 'acquisition_type', 'TeamPoints', 'TotalPoints',
        'FirstWeek', 'LastWeek',
        'Pick', 'Team', 'DraftingTeamAbbrev',
        'PointsRank', 'ValueScore'
    ]
    remaining_cols = [col for col in value_df.columns if col not in all_cols_ordered]
    display_df = value_df[all_cols_ordered + remaining_cols].rename(columns={
        'team_abbrev': 'Current Team',
        'acquisition_type': 'How Acquired',
        'DraftingTeamAbbrev': 'Drafted By (Abbrev)',
        'TeamPoints': 'Points (This Team)',
        'TotalPoints': 'Season Total Points',
        'FirstWeek': 'First Week',
        'LastWeek': 'Last Week',
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)
```

- [ ] **Step 4: Run full test suite**

```
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/dashboard.py tests/test_dashboard_display.py
git commit -m "ux: rename internal column names in Full Processed Data table"
```

---

## Task 7: Add Data Freshness Timestamp and Refresh Button to Draft Tab

**Context:** The Draft tab shows no indication of when data was last fetched. Mid-season the JSON cache goes silently stale. Users have no way to force a refresh without manually deleting files.

**Files:**
- Modify: `src/dashboard/dashboard.py`
- Modify: `src/dashboard/dashboard_logic.py`

- [ ] **Step 1: Add a helper to get data file mtimes**

Add to `dashboard_logic.py` (near the top of the file, after imports):

```python
def get_data_freshness(file_paths: list) -> Optional[str]:
    """
    Returns a human-readable string of the oldest modification time among the given files,
    or None if any file is missing.
    """
    mtimes = []
    for path in file_paths:
        if not os.path.exists(path):
            return None
        mtimes.append(os.path.getmtime(path))
    if not mtimes:
        return None
    oldest = min(mtimes)
    dt = datetime.fromtimestamp(oldest, tz=timezone.utc)
    return dt.strftime('%Y-%m-%d %H:%M UTC')
```

- [ ] **Step 2: Write a test for `get_data_freshness`**

Add to `tests/test_dashboard_logic.py`:

```python
import os
import tempfile
from src.dashboard.dashboard_logic import get_data_freshness


def test_get_data_freshness_returns_none_for_missing_file():
    assert get_data_freshness(['/nonexistent/path.json']) is None


def test_get_data_freshness_returns_string_for_existing_file():
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        path = f.name
    try:
        result = get_data_freshness([path])
        assert result is not None
        assert 'UTC' in result
    finally:
        os.unlink(path)
```

- [ ] **Step 3: Run tests to verify new tests pass**

```
pytest tests/test_dashboard_logic.py -v
```

Expected: all tests PASS (including the two new ones).

- [ ] **Step 4: Add freshness caption and refresh button in `dashboard.py`**

After Task 5, `data_loading_placeholder` no longer exists — the data loading is wrapped in `with st.spinner("Loading data..."):`. Add the freshness block **immediately after the `with st.spinner(...)` block closes**, before the `schedule_df = team_schedule_to_dataframe(...)` line:

```python
# Add import at top of file
from src.dashboard.dashboard_logic import (
    ensure_data_files_exist, process_data, plot_draft_value, team_schedule_to_dataframe,
    plot_matchup_scores_by_period, compute_duration_and_avg, get_data_freshness
)

# After data loading block, before "Proceed only if draft and stats data are loaded":
freshness = get_data_freshness([DRAFT_RESULTS_FILE, PLAYER_STATS_FILE])
col_fresh, col_refresh = st.columns([4, 1])
with col_fresh:
    if freshness:
        st.caption(f"Data last fetched: {freshness}")
    else:
        st.caption("Data freshness unknown.")
with col_refresh:
    if st.button("Refresh Data", help="Delete cached files and re-fetch from ESPN API"):
        for f in [DRAFT_RESULTS_FILE, PLAYER_STATS_FILE, TEAM_MAPPING_FILE, TEAM_SCHEDULE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        st.rerun()
```

- [ ] **Step 5: Run full test suite**

```
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/dashboard/dashboard.py src/dashboard/dashboard_logic.py tests/test_dashboard_logic.py
git commit -m "feat: add data freshness timestamp and refresh button to Draft tab"
```

---

## Task 8: Simplify `ensure_data_files_exist` Parameter List

**Context:** `ensure_data_files_exist` takes 14 positional parameters, most of which are constants from the data-processing modules. The call site in `dashboard.py` is unreadable. Grouping the ESPN credentials into the existing `config` dict and reading module constants inside the function body reduces the signature to 6 meaningful parameters.

**Files:**
- Modify: `src/dashboard/dashboard_logic.py`
- Modify: `src/dashboard/dashboard.py`

- [ ] **Step 1: Refactor the function signature in `dashboard_logic.py`**

Open `src/dashboard/dashboard_logic.py`. The current `ensure_data_files_exist` signature spans lines 15–31. Replace **only the signature and the opening docstring** — keep the entire function body (lines 36–102) verbatim:

Old signature (lines 15–31):
```python
def ensure_data_files_exist(
        config,
        draft_results_file,
        player_stats_file,
        team_mapping_file,
        team_schedule_file,
        parse_draft_results,
        fetch_box_score_stats,
        fetch_and_save_team_info,
        fetch_and_save_team_schedule,
        start_week,
        end_week,
        rate_limit_delay,
        team_info_output_dir,
        team_info_output_file,
        st=None,
):
    """
    Ensures all required data files exist, generating them if necessary.
    Returns loaded dataframes, mapping, and schedule payload.
    """
    league_id = config.get('LEAGUE_ID')
    ...
```

New signature (replace the def line + docstring only):
```python
def ensure_data_files_exist(
    config: dict,
    draft_results_file: str,
    player_stats_file: str,
    team_mapping_file: str,
    team_schedule_file: str,
    st=None,
):
    """
    Ensures all required data files exist, generating them from the ESPN API if necessary.
    Returns (draft_df, stats_df, team_map, schedule_payload).
    """
    from src.data_processing.parse_draft_results import parse_draft_results
    from src.data_processing.fetch_box_score_stats import (
        fetch_box_score_stats, START_WEEK, END_WEEK, RATE_LIMIT_DELAY
    )
    from src.data_processing.fetch_team_info import (
        fetch_and_save_team_info,
        OUTPUT_DIR as TEAM_INFO_OUTPUT_DIR,
        OUTPUT_FILE as TEAM_INFO_OUTPUT_FILE,
    )
    from src.data_processing.fetch_team_schedule import fetch_and_save_team_schedule

    league_id = config.get('LEAGUE_ID')
    year = config.get('YEAR')
    swid = config.get('SWID')
    espn_s2 = config.get('ESPN_S2')
```

Then continue with the existing body from `DATA_DIR = os.path.dirname(draft_results_file)` onward, which stays the same — the local variable names (`parse_draft_results`, `fetch_box_score_stats`, etc.) now come from the local imports instead of parameters.

Also rename the remaining parameter references in the body:
- `start_week` → `START_WEEK`
- `end_week` → `END_WEEK`
- `rate_limit_delay` → `RATE_LIMIT_DELAY`
- `team_info_output_dir` → `TEAM_INFO_OUTPUT_DIR`
- `team_info_output_file` → `TEAM_INFO_OUTPUT_FILE`

- [ ] **Step 2: Update the call site in `dashboard.py`**

First, add `TEAM_SCHEDULE_FILE` as a constant in `dashboard.py` (it was previously imported from `fetch_team_schedule`). In the `# --- Configuration ---` section, add:

```python
TEAM_SCHEDULE_FILE = 'src/data/team_schedule.json'
```

Then replace the 14-argument call with:

```python
with st.spinner("Loading data..."):
    draft_df, stats_df, team_map, schedule_payload = ensure_data_files_exist(
        config,
        DRAFT_RESULTS_FILE,
        PLAYER_STATS_FILE,
        TEAM_MAPPING_FILE,
        TEAM_SCHEDULE_FILE,
        st=st,
    )
```

Remove the now-unused imports from `dashboard.py`:

```python
# REMOVE these lines:
from src.data_processing.fetch_box_score_stats import fetch_box_score_stats, START_WEEK, END_WEEK, RATE_LIMIT_DELAY
from src.data_processing.fetch_team_info import fetch_and_save_team_info, OUTPUT_DIR as TEAM_INFO_OUTPUT_DIR, OUTPUT_FILE as TEAM_INFO_OUTPUT_FILE
from src.data_processing.fetch_team_schedule import fetch_and_save_team_schedule, OUTPUT_FILE as TEAM_SCHEDULE_FILE
from src.data_processing.parse_draft_results import parse_draft_results
```

(The `TEAM_SCHEDULE_FILE` value it imported was `'src/data/team_schedule.json'` — confirm this matches by checking `fetch_team_schedule.py`'s `OUTPUT_FILE` constant before deleting the import.)

- [ ] **Step 3: Run full test suite**

```
pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Run the app to verify it loads correctly**

```bash
streamlit run src/dashboard/dashboard.py
```

Expected: dashboard loads without ImportError or TypeError.

- [ ] **Step 5: Commit**

```bash
git add src/dashboard/dashboard.py src/dashboard/dashboard_logic.py
git commit -m "refactor: simplify ensure_data_files_exist to 6 params, move constants inside"
```

---

## Summary of Changes

| # | Type | Impact |
|---|------|--------|
| 1 | Bug fix | AvgPointsPerWeek now correct (inclusive Duration) |
| 2 | Cleanup | ~120 lines of dead code removed from `dashboard.py` |
| 3 | Bug fix | Exceptions from data processing are no longer swallowed |
| 4 | UX fix | Waiver team selector uses full names consistently |
| 5 | Bug fix | Removed blocking `time.sleep` from UI thread |
| 6 | UX | Human-readable column names in Full Data table |
| 7 | Feature | Data freshness + one-click refresh in Draft tab |
| 8 | Refactor | `ensure_data_files_exist` reduced from 14 → 6 params |
