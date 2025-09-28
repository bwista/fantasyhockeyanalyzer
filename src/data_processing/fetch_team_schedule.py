import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from espn_api.hockey import League

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

OUTPUT_DIR = "src/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "team_schedule.json")
CONFIG_FILE = "../../user_config.json"


def _determine_result_label(winner: str, side: str, team_score: Optional[float], opp_score: Optional[float]) -> str:
    winner = (winner or "").upper()
    if winner == "HOME":
        return "W" if side == "home" else "L"
    if winner == "AWAY":
        return "W" if side == "away" else "L"
    if winner == "TIE":
        return "T"
    if winner in {"UNDECIDED", "NONE", ""}:
        if team_score is not None and opp_score is not None:
            if team_score > opp_score:
                return "W"
            if team_score < opp_score:
                return "L"
            return "T"
        return "TBD"
    return "TBD"


def _is_playoff_matchup(playoff_tier: Optional[str]) -> bool:
    if playoff_tier is None:
        return False
    playoff_tier = playoff_tier.upper()
    if playoff_tier in {"NONE", "REGULAR_SEASON"}:
        return False
    return True


def fetch_and_save_team_schedule(league_id, year, swid, espn_s2, output_dir, output_file):
    """
    Fetches the full league schedule and writes a normalized per-team schedule JSON file.

    Returns:
        bool: True when the file is written successfully, False otherwise.
    """
    logging.info(f"Attempting to connect to ESPN Hockey League ID: {league_id} for year: {year}...")
    try:
        league = League(league_id=league_id, year=year, swid=swid, espn_s2=espn_s2)
        logging.info("Successfully connected to the league.")
    except Exception as exc:
        logging.error(f"Error connecting to the ESPN league: {exc}")
        logging.error("Please ensure the LEAGUE_ID, YEAR, SWID, and ESPN_S2 values are correct.")
        return False

    try:
        schedule_payload = league.espn_request.league_get(params={"view": "mMatchup"})
    except Exception as exc:
        logging.error(f"Failed to fetch league schedule: {exc}")
        return False

    schedule_items = schedule_payload.get("schedule", [])
    if not schedule_items:
        logging.warning("League schedule payload did not contain any schedule items.")

    team_lookup: Dict[int, Dict[str, Any]] = {}
    team_schedules: Dict[int, list] = {}
    for team in league.teams:
        team_lookup[team.team_id] = {
            "name": team.team_name,
            "abbrev": team.team_abbrev,
            "divisionId": team.division_id
        }
        team_schedules[team.team_id] = []

    for matchup in schedule_items:
        matchup_period = matchup.get("matchupPeriodId")
        matchup_id = matchup.get("id") or matchup.get("matchupId")
        playoff_tier = matchup.get("playoffTierType")
        winner = matchup.get("winner")
        status = matchup.get("status", {})

        home_data = matchup.get("home", {}) or {}
        away_data = matchup.get("away", {}) or {}

        for side, side_data, opp_data in (
            ("home", home_data, away_data),
            ("away", away_data, home_data),
        ):
            team_id = side_data.get("teamId")
            if team_id is None:
                continue

            opponent_id = opp_data.get("teamId") if isinstance(opp_data, dict) else None
            opponent_info = team_lookup.get(opponent_id)

            team_score = side_data.get("totalPoints")
            opp_score = opp_data.get("totalPoints") if isinstance(opp_data, dict) else None

            entry = {
                "matchupPeriod": matchup_period,
                "matchupId": matchup_id,
                "homeAway": side,
                "isPlayoff": _is_playoff_matchup(playoff_tier),
                "playoffTierType": playoff_tier,
                "opponentTeamId": opponent_id,
                "opponentTeamName": opponent_info["name"] if opponent_info else None,
                "opponentTeamAbbrev": opponent_info["abbrev"] if opponent_info else None,
                "teamScore": team_score,
                "opponentScore": opp_score,
                "result": _determine_result_label(winner, side, team_score, opp_score),
                "winner": winner,
                "isBye": opponent_id is None,
                "pointsByScoringPeriod": side_data.get("pointsByScoringPeriod", {}),
                "opponentPointsByScoringPeriod": opp_data.get("pointsByScoringPeriod", {}) if isinstance(opp_data, dict) else {},
                "status": status,
            }
            team_schedules.setdefault(team_id, []).append(entry)

    output_payload = {
        "leagueId": league.league_id,
        "seasonId": schedule_payload.get("seasonId", year),
        "generatedAt": datetime.utcnow().isoformat() + "Z",
        "currentMatchupPeriod": getattr(league, "currentMatchupPeriod", None),
        "currentScoringPeriod": getattr(league, "current_week", None),
        "teams": []
    }

    for team in league.teams:
        entries = team_schedules.get(team.team_id, [])
        entries.sort(key=lambda item: (
            item.get("matchupPeriod") if item.get("matchupPeriod") is not None else 0,
            item.get("matchupId") if item.get("matchupId") is not None else 0
        ))
        output_payload["teams"].append({
            "teamId": team.team_id,
            "teamName": team.team_name,
            "teamAbbrev": team.team_abbrev,
            "divisionId": team.division_id,
            "schedule": entries
        })

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        logging.error(f"Failed to create output directory '{output_dir}': {exc}")
        return False

    try:
        with open(output_file, "w") as handle:
            json.dump(output_payload, handle, indent=2)
        logging.info(f"Team schedule saved to: {output_file}")
        return True
    except Exception as exc:
        logging.error(f"Failed to write team schedule to {output_file}: {exc}")
        return False


def load_config():
    """Load configuration from Streamlit secrets when available, otherwise from a JSON file."""
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            config = {
                "LEAGUE_ID": st.secrets["LEAGUE_ID"],
                "YEAR": st.secrets["YEAR"],
                "SWID": st.secrets["SWID"],
                "ESPN_S2": st.secrets["ESPN_S2"],
            }
            logging.info("Loaded configuration from Streamlit secrets.")
            return config
    except (ImportError, KeyError, AttributeError):
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(script_dir, CONFIG_FILE))

    try:
        with open(config_path, "r") as handle:
            config = json.load(handle)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON configuration at {config_path}")
    except Exception as exc:
        logging.error(f"Unexpected error while loading configuration: {exc}")
    return None


if __name__ == "__main__":
    configuration = load_config()
    if not configuration:
        logging.error("No configuration available. Cannot fetch team schedule.")
        raise SystemExit(1)

    league_id = configuration.get("LEAGUE_ID")
    year = configuration.get("YEAR")
    swid = configuration.get("SWID")
    espn_s2 = configuration.get("ESPN_S2")

    if not all([league_id, year, swid, espn_s2]):
        logging.error("Missing one or more required configuration keys: LEAGUE_ID, YEAR, SWID, ESPN_S2.")
        raise SystemExit(1)

    success = fetch_and_save_team_schedule(
        league_id=league_id,
        year=year,
        swid=swid,
        espn_s2=espn_s2,
        output_dir=OUTPUT_DIR,
        output_file=OUTPUT_FILE
    )

    if success:
        logging.info("Team schedule fetch completed successfully.")
    else:
        logging.error("Team schedule fetch failed.")
