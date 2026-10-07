# Data wrangling for Linear Regression 2.1 (goal difference).
# Reads the input_*.csv files in this folder and builds LR_2.1_dataset_clean.csv.

import pandas as pd

SRC = "input_"

# load the input files
matches = pd.read_csv(SRC + "matches_all.csv")
rank    = pd.read_csv(SRC + "rankings_jun2026.csv").set_index("team")["fifa_rank_jun2026"]
form    = pd.read_csv(SRC + "form.csv").set_index("team")
host    = pd.read_csv(SRC + "host_flag.csv").set_index("team")["host_flag"]
value   = pd.read_csv(SRC + "market_value.csv").set_index("team")["squad_value_eur_million"]
age     = pd.read_csv(SRC + "squad_age.csv").set_index("team")["avg_age"]
h2h     = pd.read_csv(SRC + "h2h_raw.csv").set_index("match_id")["h2h_avg_gd"]
venues  = pd.read_csv(SRC + "match_venues.csv").set_index("match_id")
alt     = pd.read_csv(SRC + "stadium_altitude.csv").set_index("stadium")["altitude_m"]

# check the inputs
assert len(matches) == 104
teams = set(matches.team_a) | set(matches.team_b)
assert len(teams) == 48
for name, s in [("rank", rank), ("host", host), ("value", value), ("age", age), ("form", form["avg_gf"])]:
    assert not (teams - set(s.index)), f"{name} is missing some teams"
assert set(matches.match_id) <= set(h2h.index)

# build the dataset (team_a minus team_b for each variable)
a, b = matches["team_a"], matches["team_b"]
stadium = matches["match_id"].map(venues["stadium"])

df = pd.DataFrame({
    # labels
    "match_id": matches["match_id"],
    "stage":    matches["stage"],
    "team_a":   a,
    "team_b":   b,
    "stadium":  stadium,
    # what we predict
    "goal_difference": matches["score_a"] - matches["score_b"],
    # the 8 explanatory variables
    "fifa_rank_diff":         b.map(rank) - a.map(rank),
    "attack_diff":            (a.map(form["avg_gf"]) - b.map(form["avg_gf"])).round(3),
    "defence_diff":           (a.map(form["avg_ga"]) - b.map(form["avg_ga"])).round(3),
    "h2h_avg_gd":             matches["match_id"].map(h2h),
    "squad_value_diff_eur_m": (a.map(value) - b.map(value)).round(2),
    "squad_age_diff":         (a.map(age) - b.map(age)).round(2),
    "venue_altitude_m":       stadium.map(alt),
    "host_advantage_diff":    a.map(host) - b.map(host),
})

# check the output
explanatory = ["fifa_rank_diff", "attack_diff", "defence_diff", "h2h_avg_gd",
               "squad_value_diff_eur_m", "squad_age_diff", "venue_altitude_m",
               "host_advantage_diff"]
assert df.shape[0] == 104 and len(explanatory) == 8
assert df[explanatory].isna().sum().sum() == 0

df.to_csv("LR_2.1_dataset_clean.csv", index=False)
print("saved", df.shape)
