# Fantasy Hockey Analyzer

A data analysis tool for ESPN fantasy hockey leagues. This application fetches player stats, draft results, and team information to provide insights into draft value and player acquisitions via a Streamlit dashboard.

## Features

*   **Data Fetching:**
    *   Retrieves detailed player statistics (box scores).
    *   Fetches draft results for the league.
    *   Gathers team information (names, abbreviations).
*   **Interactive Dashboard (`src/dashboard/dashboard.py`):**
    *   Visualizes draft value by plotting Overall Pick vs. Fantasy Points Rank.
    *   Identifies best and worst value draft picks.
    *   Provides team-specific draft value analysis.
    *   Analyzes waiver wire and trade acquisitions, highlighting top performers.
    *   Displays aggregated player and team statistics.
    *   Allows users to filter and explore data.
*   **Configuration:**
    *   Uses a `user_config.json` file for ESPN league credentials and settings.

## Prerequisites

*   Python 3.8+
*   pip (Python package installer)

## Setup and Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd fantasyhockeyanalyzer
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    # venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` is up-to-date with all necessary libraries like `streamlit`, `pandas`, `espn_api`, `plotly`, `statsmodels`, `numpy`)*

## Configuration

Before running the application, you need to configure your ESPN league details:

1.  **Create `user_config.json` in the root directory of the project.**
2.  **Add your league details to this file. It should look like this:**

    ```json
    {
      "LEAGUE_ID": 12345678,
      "YEAR": 2025,
      "SWID": "{YOUR_SWID_COOKIE_VALUE}",
      "ESPN_S2": "YOUR_ESPN_S2_COOKIE_VALUE"
    }
    ```

    *   Replace `12345678` with your actual ESPN League ID.
    *   Replace `2025` with the fantasy season year you want to analyze.
    *   Replace `{YOUR_SWID_COOKIE_VALUE}` with your ESPN SWID cookie.
    *   Replace `YOUR_ESPN_S2_COOKIE_VALUE` with your ESPN_S2 cookie.

    **How to find your SWID and ESPN_S2 cookies:**
    *   Log in to your ESPN fantasy league in a web browser.
    *   Open your browser's developer tools (usually by pressing F12).
    *   Go to the "Application" (Chrome/Edge) or "Storage" (Firefox) tab.
    *   Find the cookies for `espn.com` or `fantasy.espn.com`.
    *   Locate the `SWID` and `espn_s2` cookies and copy their values.

## How to Run the Application

1.  **Ensure your `user_config.json` is correctly set up.**
2.  **Navigate to the project's root directory in your terminal.**
3.  **Run the Streamlit dashboard:**
    ```bash
    streamlit run src/dashboard/dashboard.py
    ```
4.  The application will open in your web browser.
    *   On the first run, or if data files (`src/data/*.json`) are missing, the dashboard will attempt to fetch the necessary data from the ESPN API using the credentials in `user_config.json`. This might take a few moments.
    *   Subsequent runs will load data from the local files if they exist.

## Project Structure

```
fantasyhockeyanalyzer/
├── .gitignore
├── PLAN.MD
├── README.md
├── requirements.txt
├── user_config.json  # User-specific ESPN league configuration
├── src/
│   ├── __init__.py
│   ├── analysis/       # (Potential future location for more complex analysis scripts)
│   │   └── calculate_scoring_system.py
│   ├── dashboard/      # Streamlit dashboard code
│   │   ├── __init__.py
│   │   └── dashboard.py
│   ├── data/           # Stores fetched data (JSON files)
│   │   ├── __init__.py
│   │   ├── box_score_stats.json
│   │   ├── draft_results.json
│   │   └── team_mapping.json
│   ├── data_processing/ # Scripts for fetching and processing data
│   │   ├── __init__.py
│   │   ├── fetch_box_score_stats.py
│   │   ├── fetch_player_stats.py
│   │   ├── fetch_team_info.py
│   │   └── parse_draft_results.py
│   └── utils/          # (Potential future location for utility functions)
│       └── __init__.py
└── venv/               # Virtual environment (if created)
```

## License

MIT