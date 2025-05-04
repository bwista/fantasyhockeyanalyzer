# Plan: Enhance Fantasy Hockey Dashboard with Acquisition Analysis

This plan outlines the steps to modify `src/dashboard/dashboard.py` to incorporate analysis of how players were acquired (drafted or waiver/trade), track their team affiliations throughout the season, and add a dedicated section for "Best Waiver/Trade Acquisitions".

**Definition of Acquisition Types:**

*   **`drafted`**:
    *   The player's *first* recorded team affiliation in the season stats (based on `FirstWeek`) matches the team that drafted them (according to `draft_results.json` and `team_mapping.json`).
*   **`waiver`**:
    *   The player was *not* listed in `draft_results.json`.
    *   Any team affiliation record for a player *after* their initial 'drafted' record, indicating a team change post-draft.

## I. Data Loading and Processing Modifications

1.  **Load Data:** Maintain current loading logic for `draft_df` (Draft Results), `stats_df` (Weekly Box Score Stats), and `team_map` (Team Name Mapping).
2.  **Aggregate Stats per Player per Team:**
    *   Group `stats_df` by `name` and `team_abbrev` to get total points accumulated by each player *for each team* they played on. Keep track of the first and last weeks played for ordering and duration context.
    *   Example: `player_team_stats = stats_df.groupby(['name', 'team_abbrev']).agg(TeamPoints=('total_points', 'sum'), FirstWeek=('week', 'min'), LastWeek=('week', 'max')).reset_index()`
3.  **Aggregate Overall Player Stats:**
    *   Separately, group `stats_df` by `name` only to get the *total* points for the entire season across all teams.
    *   Example: `total_player_stats = stats_df.groupby('name').agg(TotalPoints=('total_points', 'sum')).reset_index()`
4.  **Prepare Draft Data:**
    *   Add `DraftingTeamAbbrev` to `draft_df` using `team_map`.
    *   Example: `draft_df['DraftingTeamAbbrev'] = draft_df['Team'].map(team_map)`
    *   Select relevant columns: `['Player', 'Overall Pick', 'DraftingTeamAbbrev']`. Rename `Player` to `name` for consistency.
5.  **Merge Draft Info with Per-Team Stats:**
    *   Perform a `left` merge of `draft_df` onto `player_team_stats` using `name`. This adds draft information to each player-team record. Records for undrafted players will have NaN in draft columns.
    *   Example: `merged_team_stats = pd.merge(player_team_stats, draft_df[['name', 'Overall Pick', 'DraftingTeamAbbrev']], on='name', how='left')`
6.  **Determine Acquisition Type:**
    *   Sort `merged_team_stats` by `name` and `FirstWeek` to process team affiliations chronologically for each player.
    *   Iterate or use group-wise operations to apply the `acquisition_type` logic:
        *   Initialize `acquisition_type` column.
        *   For each player (`name`):
            *   Find the record with the minimum `FirstWeek`.
            *   If `Overall Pick` is not NaN AND `team_abbrev` for this first record matches `DraftingTeamAbbrev`, set `acquisition_type` to `'drafted'`.
            *   If `Overall Pick` is NaN, set `acquisition_type` to `'waiver'`.
            *   If `Overall Pick` is not NaN but `team_abbrev` does *not* match `DraftingTeamAbbrev`, set `acquisition_type` to `'waiver'` (implies drafted but immediately traded/dropped before playing, or data inconsistency).
            *   For any subsequent records for the *same player* (later `FirstWeek`), set `acquisition_type` to `'waiver'`.
    *   This results in `final_team_stats_df` with columns like `name`, `team_abbrev`, `TeamPoints`, `FirstWeek`, `LastWeek`, `Overall Pick`, `DraftingTeamAbbrev`, `acquisition_type`.
7.  **Combine with Overall Stats:**
    *   Merge `total_player_stats` into `final_team_stats_df` based on `name` to add the overall `TotalPoints` for ranking purposes.
    *   Example: `final_df = pd.merge(final_team_stats_df, total_player_stats, on='name', how='left')`

## II. Value and Ranking Calculation Adjustments

1.  **Calculate `PointsRank`:** Calculate based on overall `TotalPoints` across all players (using `total_player_stats` before the final merge might be easier). Add this rank back to `final_df`.
2.  **Calculate `ValueScore`:**
    *   This score primarily applies to the *draft* decision. Calculate `ValueScore = DraftRank - PointsRank` (where `DraftRank` is `Overall Pick`).
    *   Assign this score only to the player's initial 'drafted' record in `final_df` (identified by `FirstWeek` and `acquisition_type == 'drafted'`). Mark as NaN for waiver acquisitions/records.

## III. Add "Best Waiver/Trade Acquisitions" Section

1.  **Filter Acquisitions:** Create `acquisitions_df` by filtering `final_df` for records where `acquisition_type == 'waiver'`. Since a player might have multiple 'waiver' records if traded multiple times, ensure you get a unique list of *players* acquired via waiver/trade.
2.  **Identify Best Acquisitions:**
    *   Rank the unique players identified in the previous step based on their overall season `TotalPoints`.
3.  **Sort Acquisitions:** Sort the unique list of acquired players by `TotalPoints` descending.
4.  **Create New Streamlit Section:**
    *   Add header: `st.header("Waiver Wire & Trade Analysis (Best Acquisitions)")`.
    *   Display Top N Overall Acquisitions table (unique players): `name`, `TotalPoints`, `Overall Pick` (if applicable, otherwise NaN/blank), `DraftingTeamAbbrev` (if applicable).
    *   Display Top N Team Acquisitions table: Filter `final_df` for `acquisition_type == 'waiver'`, group by `team_abbrev` (the acquiring team), and show the top players acquired by each team based on `TeamPoints` scored *for that team*. Columns: `name`, `TeamPoints`.

## IV. Update Existing Displays

1.  **Scatter Plot (Draft Value):**
    *   Filter `final_df` to include only the *initial drafted record* for each player (`acquisition_type == 'drafted'` and identified using `FirstWeek`). Plot `TotalPoints` vs `Overall Pick`.
2.  **Value Tables (Best/Worst Draft Value):**
    *   Use the 'drafted' records from `final_df` (as filtered above). Calculate `ValueScore` on these records and sort.
3.  **Team-Specific Value:**
    *   Show `ValueScore` for players *drafted* by each team. Filter `final_df` for `acquisition_type == 'drafted'` and group/filter by `DraftingTeamAbbrev`.
4.  **Full Data Table:**
    *   Display `final_df`. Include columns like `name`, `team_abbrev`, `TeamPoints`, `TotalPoints`, `acquisition_type`, `Overall Pick`, `DraftingTeamAbbrev`, `FirstWeek`, `LastWeek`. Allow sorting and filtering.

## V. Mermaid Diagram (Updated Logic)

```mermaid
graph TD
    subgraph Data Loading
        A[Load draft_results.json] --> P1;
        B[Load box_score_stats.json] --> P1;
        C[Load team_mapping.json] --> P1;
    end

    subgraph Initial Processing
        P1 --> D[Map DraftingTeamAbbrev to draft_df];
        P1 --> E[Aggregate stats_df by name, team_abbrev -> player_team_stats (TeamPoints, FirstWeek, LastWeek)];
        P1 --> F[Aggregate stats_df by name -> total_player_stats (TotalPoints)];
    end

    subgraph Merging & Acquisition Logic
        E --> G{Left Merge: player_team_stats + draft_df};
        D --> G;
        G --> H[Sort by name, FirstWeek];
        H --> I[Calculate acquisition_type ('drafted'/'waiver') per record];
        I --> J{Final Team Stats DF};
    end

    subgraph Final Data Assembly
        J --> K{Left Merge: Final Team Stats + total_player_stats};
        F --> K;
        K --> L[Calculate PointsRank (based on TotalPoints)];
        L --> M[Calculate ValueScore (DraftRank - PointsRank, for 'drafted' records)];
        M --> N{Final DataFrame (final_df)};
    end

    subgraph Display Updates
        N -- Filter: acquisition_type=='drafted' --> O[Display: Scatter Plot (TotalPoints vs DraftPick)];
        N -- Filter: acquisition_type=='drafted' --> P[Display: Best/Worst Value Tables (ValueScore)];
        N -- Filter: acquisition_type=='drafted' --> Q[Display: Team-Specific Value (Drafting Team)];
        N --> R[Display: Full Data Table (All Records)];
    end

    subgraph New Acquisition Section
        N -- Filter: acquisition_type=='waiver' --> S[Filter: Waiver Acquisitions (Unique Players)];
        S --> T[Sort Unique Players by TotalPoints];
        T --> U[Display: Top Overall Acquisitions (by TotalPoints)];
        N -- Filter: acquisition_type=='waiver' --> V[Filter: Waiver Records];
        V --> W[Sort Waiver Records by TeamPoints];
        W --> X[Display: Top Team Acquisitions (by TeamPoints for that team)];
    end

    style E fill:#f9f,stroke:#333,stroke-width:2px
    style I fill:#f9f,stroke:#333,stroke-width:2px
    style M fill:#f9f,stroke:#333,stroke-width:2px
    style S fill:#ccf,stroke:#333,stroke-width:2px
    style T fill:#ccf,stroke:#333,stroke-width:2px
    style U fill:#ccf,stroke:#333,stroke-width:2px
    style V fill:#ccf,stroke:#333,stroke-width:2px
    style W fill:#ccf,stroke:#333,stroke-width:2px
    style X fill:#ccf,stroke:#333,stroke-width:2px