# Data wrangling for Linear Regression 2.2 (goals scored). Sources: FBref and fifa.com.
from pathlib import Path

import pandas as pd

FOLDER = Path(__file__).parent
RAW = FOLDER / "raw"
OUTPUT = FOLDER / "wc2026_team_goals_model2_2.csv"

HOSTS = ["USA", "Canada", "Mexico"]

# Best previous World Cup finish as an ordered number (higher = went further)
FINISH_CODE = {
    "Never played": 0,
    "Group stage": 1,      # also "first round" in older tournaments
    "Round of 16": 2,      # also "second round"
    "Quarter-final": 3,
    "Semi-final": 4,       # finished 3rd or 4th
    "Runner-up": 5,
    "Winner": 6,
}

# ---------------------------------------------------------------------------
# 1. Match results (FBref)
# ---------------------------------------------------------------------------
matches = pd.read_csv(RAW / "fbref_wc2026_match_results.csv")

# ---------------------------------------------------------------------------
# 2. One row per team per match
# ---------------------------------------------------------------------------
home_rows = matches.rename(columns={"home": "team", "away": "opponent",
                                    "home_goals": "goals_scored"})
home_rows["order"] = 0
away_rows = matches.rename(columns={"away": "team", "home": "opponent",
                                    "away_goals": "goals_scored"})
away_rows["order"] = 1

keep = ["match_id", "date", "stage", "team", "opponent", "goals_scored", "order"]
data = pd.concat([home_rows[keep], away_rows[keep]], ignore_index=True)
data = data.sort_values(["match_id", "order"]).drop(columns="order").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. Match-level variables
# ---------------------------------------------------------------------------
data["team_is_host"] = data["team"].isin(HOSTS).astype(int)
data["is_knockout"] = (data["stage"] != "Group stage").astype(int)

# ---------------------------------------------------------------------------
# 4. FIFA ranking (fifa.com, 11 June 2026 - published before the first match)
# ---------------------------------------------------------------------------
ranking = pd.read_csv(RAW / "fifa_ranking_2026-06-11.csv").set_index("team")["fifa_rank"]
data["team_fifa_rank"] = data["team"].map(ranking)
data["opp_fifa_rank"] = data["opponent"].map(ranking)

# ---------------------------------------------------------------------------
# 5. World Cup history before 2026 (fifa.com)
# ---------------------------------------------------------------------------
history = pd.read_csv(RAW / "wc_history_before_2026.csv").set_index("team")
history["finish_code"] = history["best_wc_finish"].map(FINISH_CODE)

data["team_prev_wc_appearances"] = data["team"].map(history["prev_wc_appearances"])
data["opp_prev_wc_appearances"] = data["opponent"].map(history["prev_wc_appearances"])
data["team_best_wc_finish"] = data["team"].map(history["finish_code"])
data["opp_best_wc_finish"] = data["opponent"].map(history["finish_code"])

# ---------------------------------------------------------------------------
# 6. Checks, then save
# ---------------------------------------------------------------------------
if data.isna().any().any():
    bad = data[data.isna().any(axis=1)]
    raise SystemExit("Missing values - check team spellings for: "
                     f"{sorted(set(bad['team']) | set(bad['opponent']))}")
assert len(data) == 208, f"Expected 208 rows, got {len(data)}"
assert data["team"].nunique() == 48, "Expected 48 teams"

columns = [
    "match_id", "date", "stage", "team", "opponent",
    "team_fifa_rank", "opp_fifa_rank", "team_is_host", "is_knockout",
    "team_prev_wc_appearances", "opp_prev_wc_appearances",
    "team_best_wc_finish", "opp_best_wc_finish",
    "goals_scored",
]
data = data[columns].astype({c: int for c in columns[5:]})
data.to_csv(OUTPUT, index=False)

print(f"Saved {OUTPUT.name}: {len(data)} rows, {data['team'].nunique()} teams, no missing values")
