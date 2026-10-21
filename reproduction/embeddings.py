"""
Loading SocialVec entity embeddings and projecting users into the embedding space.

The `socialvec` package wraps these embeddings, but importing it requires
TensorFlow. We only need the gensim Word2Vec model it ships, so we load the
same model files directly.
"""

import gzip
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODELS_DIR, SOCIALVEC_METADATA, SOCIALVEC_MODELS


def download(url: str, target: Path) -> Path:
    """Download `url` to `target` unless it already exists."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {target.name} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)
    return target


class SocialVec:
    """
    Pre-trained SocialVec embeddings of ~200K popular Twitter accounts.

    Args:
        model_name: "2020" (used for the ranking experiments) or "2020c"
            (the version the socio-demographic classifiers were trained on).
    """

    def __init__(self, model_name: str = "2020", models_dir: Path = MODELS_DIR):
        filename, url = SOCIALVEC_MODELS[model_name]
        path = download(url, models_dir / filename)
        with gzip.open(path, "rb") as f:
            model = pickle.load(f)
        self.model_name = model_name
        self.vectors = model.wv.vectors  # float32, shape (n_entities, 100)
        self.key_to_index = model.wv.key_to_index  # keys are Twitter ids as strings
        self.dim = self.vectors.shape[1]

    def __contains__(self, account_id) -> bool:
        return str(account_id) in self.key_to_index

    def indices(self, account_ids) -> np.ndarray:
        """Row index of each account id in the embedding matrix (-1 if not an entity)."""
        k2i = self.key_to_index
        return np.fromiter((k2i.get(str(a), -1) for a in account_ids), dtype=np.int64)

    def vector(self, account_id) -> np.ndarray:
        return self.vectors[self.key_to_index[str(account_id)]]

    def average(self, idx: np.ndarray) -> np.ndarray:
        """
        Average the embeddings at the given row indices, ignoring -1 (non-entity) ids.

        Mirrors `SocialVec.get_average_embeddings`: the mean is taken in float32
        and an all-zero vector is returned when no entity is found.
        """
        idx = idx[idx >= 0]
        if len(idx) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return self.vectors[idx].mean(axis=0)


def load_metadata(models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Screen name, display name and description of SocialVec entities, keyed by Twitter id."""
    filename, url = SOCIALVEC_METADATA
    path = download(url, models_dir / filename)
    return pd.read_parquet(path)
