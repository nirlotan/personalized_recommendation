"""
Socio-demographic profiles of the followers of each candidate entity (Figure 2).

Users are embedded with the SocialVec "2020c" model and classified with the
trait classifiers distributed in the `socialvec` package. Each classifier is a
single sigmoid unit stored as a Keras .h5 file; we read its weights with h5py
and apply it with NumPy, so TensorFlow is not needed.
"""

import importlib.util
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .embeddings import SocialVec
from .paper_results import RADAR_TRAITS

# trait -> (classifier file, label when sigmoid >= 0.5, label when < 0.5).
# socialvec/config.yaml lists education as (Degree, HighSchool) and age as
# (Under25, Over25), but with those labels ~98% of the followers of e.g.
# '60 Minutes' come out under 25 and without a degree. The paper's Figure 2
# reads both classifiers the other way round (see the `1 - x` education values
# in the notebook), so we swap them here.
CLASSIFIERS = {
    "political": ("SocialVec_political_2022.h5", "Republican", "Democrat"),
    "gender": ("SocialVec_gender_2022.h5", "Male", "Female"),
    "ethnicity": ("SocialVec_ethnicity_2022.h5", "Caucasian", "AfrAmerican"),
    "education": ("SocialVec_education_2022.h5", "HighSchool", "Degree"),
    "age": ("SocialVec_age_2022.h5", "Over25", "Under25"),
}
# The trait value plotted on each radar axis
RADAR_AXES = {
    "Democrat": ("political", "Democrat"),
    "Female": ("gender", "Female"),
    "Caucasian": ("ethnicity", "Caucasian"),
    "Degree": ("education", "Degree"),
    "Over-25": ("age", "Over25"),
}


def classifier_dir() -> Path:
    """Folder with the classifier weights inside the installed `socialvec` package."""
    spec = importlib.util.find_spec("socialvec")  # locates the package without importing it
    if spec is None:
        raise ImportError("The `socialvec` package (source of the classifier weights) is not installed")
    return Path(next(iter(spec.submodule_search_locations))) / "models"


class TraitClassifier:
    """A logistic-regression trait classifier loaded from a Keras .h5 file."""

    def __init__(self, path: Path, positive_label: str, negative_label: str):
        with h5py.File(path, "r") as f:
            dense = f["model_weights/dense/dense"]
            self.kernel = dense["kernel:0"][()].ravel().astype(np.float64)
            self.bias = float(dense["bias:0"][()][0])
        self.positive_label = positive_label
        self.negative_label = negative_label

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """Predicted label for each row of `vectors`."""
        p = 1.0 / (1.0 + np.exp(-(vectors.astype(np.float64) @ self.kernel + self.bias)))
        return np.where(p >= 0.5, self.positive_label, self.negative_label)


def classify_users(users: pd.DataFrame, sv: SocialVec) -> pd.DataFrame:
    """Predict every trait for every user from the average embedding of all accounts they follow."""
    folder = classifier_dir()
    vectors = np.stack([
        sv.average(sv.indices(friends))
        for friends in tqdm(users["friends_ids"], desc="Embedding users (2020c)")
    ])
    traits = pd.DataFrame(index=users.index)
    for trait, (filename, positive, negative) in CLASSIFIERS.items():
        traits[trait] = TraitClassifier(folder / filename, positive, negative).predict(vectors)
    return traits


def entity_profiles(users: pd.DataFrame, traits: pd.DataFrame,
                    categories: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For every candidate entity, the share of its followers (among the
    experiment's users) predicted to have each radar trait value.

    Returns:
        DataFrame indexed by entity id with columns screen_name, n_followers
        and one column per radar axis (RADAR_TRAITS).
    """
    follows = (users[["experiment_accounts"]].join(traits)
               .explode("experiment_accounts")
               .rename(columns={"experiment_accounts": "account_id"}))
    follows["account_id"] = follows["account_id"].astype(np.int64)
    for axis, (trait, value) in RADAR_AXES.items():
        follows[axis] = follows[trait] == value
    profiles = follows.groupby("account_id")[RADAR_TRAITS].mean()
    profiles.insert(0, "n_followers", follows.groupby("account_id").size())

    names = pd.concat(categories.values()).drop_duplicates("twitter_user_id")
    names = names.set_index("twitter_user_id")["twitter_screen_name"]
    profiles.insert(0, "screen_name", names.reindex(profiles.index).values)
    return profiles


def category_radar(profiles: pd.DataFrame, category: pd.DataFrame, screen_names) -> pd.DataFrame:
    """
    Radar values for selected entities of a category, plus 'ALL': the mean
    profile over all the category's entities.
    """
    ids = category["twitter_user_id"].values
    lookup = {s.lower(): i for s, i in zip(category["twitter_screen_name"], ids)}
    rows = {name: profiles.loc[lookup[name.lower()], RADAR_TRAITS] for name in screen_names}
    rows["ALL"] = profiles.loc[profiles.index.intersection(ids), RADAR_TRAITS].mean()
    return pd.DataFrame(rows).T.astype(float)
