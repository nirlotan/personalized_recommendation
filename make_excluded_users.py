"""
Users of the Section 5 / Tables 9-11 experiments to exclude from the analyses: organizations and likely bots.

Profiles (bio, follower, tweet and like counts, account age) come from the original collection file
data/users_selected_for_the_experiment2.pkl, whose rows match the anonymized users file one to one.
Only internal row ids are written; the profile data stays in data/.

    python make_excluded_users.py

Rules:
    organization          - found by an organization keyword search of bios and handles (154 matches),
                            then confirmed by reading each bio (ORG_ROWS, POSSIBLE_ORG_ROWS)
    bot: very active      - more than 144 tweets a day on average over the account's life
    bot: 2+ weak signals  - at least two of: more than 72 tweets a day, following 4,900+ accounts, followers
                            under 5% of follows (with 1,000+ follows), a handle ending in 6+ digits with an
                            empty bio, no likes, an account under 90 days old

Writes results/excluded_users.csv (user row, reason).
"""

import pandas as pd

from reproduction.config import DATA_DIR, RESULTS_DIR

ORG_ROWS = [3437, 4673, 5291, 5465, 5584, 5828, 6182, 6530, 6638, 6645, 6814, 7255, 7874, 8009, 8457, 9770, 10513,
            12844, 13299, 13593, 13695]
POSSIBLE_ORG_ROWS = [4077, 8107, 13393]


def main():
    users = pd.read_csv(RESULTS_DIR / "llm" / "band_per_user_n200.csv.gz", usecols=["user"])["user"]
    p = pd.read_pickle(DATA_DIR / "users_selected_for_the_experiment2.pkl").reset_index(drop=True).loc[users]
    for c in ["followers_count", "friends_count", "statuses_count", "favourites_count"]:
        p[c] = pd.to_numeric(p[c], errors="coerce")
    age = (pd.to_datetime(p["collected_at"], utc=True) - pd.to_datetime(p["created_at"], utc=True)).dt.days.clip(lower=1)
    per_day = p["statuses_count"] / age
    bio = p["description"].fillna("").str.strip()
    weak = ((per_day > 72).astype(int) + (p["friends_count"] >= 4900)
            + ((p["followers_count"] < 0.05 * p["friends_count"]) & (p["friends_count"] >= 1000))
            + (p["screen_name"].str.contains(r"\d{6,}$") & (bio == "")) + (p["favourites_count"] == 0) + (age < 90))

    reasons = {}
    for row in p.index:
        if row in ORG_ROWS:
            reasons[row] = "organization"
        elif row in POSSIBLE_ORG_ROWS:
            reasons[row] = "possible organization"
        elif per_day[row] > 144:
            reasons[row] = "bot: very active"
        elif weak[row] >= 2:
            reasons[row] = "bot: 2+ weak signals"
    out = pd.DataFrame({"user": list(reasons), "reason": list(reasons.values())}).sort_values("user")
    out.to_csv(RESULTS_DIR / "excluded_users.csv", index=False)
    print(f"{len(out)} of {len(p)} users excluded:", out["reason"].value_counts().to_dict())


if __name__ == "__main__":
    main()
