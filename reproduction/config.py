"""
Paths, constants and category metadata shared by the reproduction code.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CATEGORIES_XLSX = DATA_DIR / "popular_accounts_manually_validated.xlsx"
# The anonymized pickle has the same rows (same order, same follow lists) as
# users_selected_for_the_experiment2.pkl, without the users' personal fields.
USERS_PICKLE = DATA_DIR / "users_selected_for_the_experiment_anonymized.pickle"

# Download locations used by the `socialvec` package (config.yaml).
SOCIALVEC_MODELS = {
    "2020": (
        "SocialVec2020.pkl.gz",
        "https://www.dropbox.com/s/lti8npc8r2l4wch/SocialVec2020.pkl.gz?dl=1",
    ),
    "2020c": (
        "SocialVec2020c.pkl.gz",
        "https://www.dropbox.com/scl/fi/1s1kay2z4a430lkz1erjn/SocialVec2020c.pkl.gz"
        "?rlkey=fcfbb83gkdb80ghivzt62ljmk&dl=1",
    ),
}
SOCIALVEC_METADATA = (
    "metadata.parquet.gz",
    "https://www.dropbox.com/s/060xprg4emblqwu/metadata.parquet.gz?dl=1",
)

# Experimental protocol (as implemented in legacy_code/recommendation_p2.ipynb)
SEED = 76
MIN_EXPERIMENT_ACCOUNTS = 10  # keep users following *more than* this many experiment entities
SE1_SAMPLE_SIZES = [10, 20, 30, 40, 50, 80, 100, 300, 500, 5000]
SE2_SAMPLE_SIZES = [10, 20, 30, 40, 50]
GRID_CATEGORIES = [1, 2, 3, 4, 5]          # Table 4 rows (k)
GRID_ACCOUNTS_PER_CATEGORY = [1, 2, 3, 4, 5, 6, 7]  # Table 4 columns (n)

# Sheet name in the spreadsheet -> label used in the paper, in the paper's table order.
CATEGORY_LABELS = {
    "MusicalArtist": "Musical artists",
    "news": "News outlets",
    "Comedian": "Comedians",
    "Politician": "Politicians",
    "TelevisionStation": "TV stations",
    "Actor": "Actors",
    "TelevisionShow": "TV shows",
    "SportsTeam": "Sports teams",
    "fashion": "Fashion",
    "Journalist": "Journalists",
    "TelevisionHost": "TV hosts",
    "Film": "Films",
    "Resturants": "Food chains",
    "cars": "Car makers",
}
PAPER_CATEGORY_ORDER = list(CATEGORY_LABELS)
