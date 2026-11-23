"""
Numbers published in the paper (arXiv:2603.10148v1), used to check the reproduction.

Figure 2 is only available as an image; its values are taken from the radar-plot
cells of legacy_code/recommendation_p2.ipynb, which produced that figure.
"""

import pandas as pd

from .config import PAPER_CATEGORY_ORDER

# Table 2: entities per user, popularity MAP, social embeddings 'All' MAP, 'Sampled' MAP
TABLE2 = pd.DataFrame(
    [
        [7.6, 0.520, 0.633, 0.591],
        [8.2, 0.581, 0.692, 0.661],
        [6.8, 0.545, 0.599, 0.594],
        [11.2, 0.720, 0.777, 0.755],
        [4.5, 0.469, 0.618, 0.603],
        [6.6, 0.521, 0.647, 0.602],
        [3.9, 0.365, 0.635, 0.551],
        [4.3, 0.349, 0.510, 0.385],
        [2.6, 0.372, 0.447, 0.373],
        [10.1, 0.622, 0.675, 0.640],
        [1.8, 0.301, 0.428, 0.394],
        [1.4, 0.221, 0.399, 0.241],
        [5.2, 0.492, 0.499, 0.481],
        [5.7, 0.472, 0.437, 0.434],
    ],
    index=PAPER_CATEGORY_ORDER,
    columns=["n_relevant", "popularity", "all", "sampled"],
)
TABLE2_OVERALL = {"popularity": 0.468, "all": 0.571, "sampled": 0.522}

# Section 4.3 / Figure 1: overall MAP for k sampled entities
FIGURE1_SE1 = {10: 0.532, 20: 0.548, 50: 0.562}
FIGURE1_SE2 = {10: 0.517, 20: 0.521}

# Table 3: 'Social' column (SE1 with k = 50) and the GPT-4o columns
TABLE3 = pd.DataFrame(
    [
        [0.625, 0.508, 0.582, 0.611, 0.667],
        [0.688, 0.648, 0.695, 0.698, 0.716],
        [0.593, 0.576, 0.638, 0.643, 0.655],
        [0.774, 0.785, 0.797, 0.801, 0.811],
        [0.605, 0.507, 0.636, 0.640, 0.664],
        [0.637, 0.511, 0.605, 0.603, 0.662],
        [0.622, 0.427, 0.595, 0.612, 0.637],
        [0.495, 0.347, 0.393, 0.391, 0.571],
        [0.443, 0.377, 0.409, 0.409, 0.493],
        [0.668, 0.680, 0.673, 0.675, 0.704],
        [0.422, 0.313, 0.436, 0.451, 0.522],
        [0.372, 0.216, 0.306, 0.308, 0.399],
        [0.485, 0.492, 0.496, 0.504, 0.496],
        [0.438, 0.472, 0.480, 0.487, 0.409],
    ],
    index=PAPER_CATEGORY_ORDER,
    columns=["social_k50", "llm_baseline", "llm_k12", "llm_k25", "llm_k50"],
)

# Table 4: rows k = 1..5 categories, columns n = 1..7 entities per category
TABLE4 = pd.DataFrame(
    [
        [0.471, 0.478, 0.482, 0.485, 0.485, 0.485, 0.485],
        [0.476, 0.490, 0.496, 0.500, 0.501, 0.501, 0.502],
        [0.476, 0.493, 0.503, 0.506, 0.509, 0.511, 0.511],
        [0.477, 0.493, 0.500, 0.508, 0.511, 0.516, 0.516],
        [0.477, 0.495, 0.507, 0.512, 0.515, 0.517, 0.518],
    ],
    index=pd.Index([1, 2, 3, 4, 5], name="k categories"),
    columns=pd.Index([1, 2, 3, 4, 5, 6, 7], name="n per category"),
)

# Figure 2 radar values: share of followers classified as
# [Democrat, Female, Caucasian, Degree, Over-25]
RADAR_TRAITS = ["Democrat", "Female", "Caucasian", "Degree", "Over-25"]
FIGURE2 = {
    "Musical artists": {
        "taylorswift13": [0.840, 0.507, 0.919, 1 - 0.384, 0.815],
        "kanyewest": [0.779, 0.330, 0.792, 1 - 0.488, 0.760],
        "kendricklamar": [0.905, 0.372, 0.697, 1 - 0.614, 0.679],
        "cher": [0.908, 0.555, 0.937, 1 - 0.192, 0.930],
        "ALL": [0.865, 0.467, 0.828, 1 - 0.477, 0.762],
    },
    "Car makers": {
        "Ford": [0.633, 0.271, 0.923, 0.587, 0.92],
        "Rivian": [0.803, 0.115, 0.955, 0.752, 0.949],
        "Jeep": [0.658, 0.315, 0.908, 0.483, 0.86],
        "Ferrari": [0.673, 0.126, 0.857, 0.497, 0.844],
        "ALL": [0.697, 0.222, 0.876, 0.532, 0.892],
    },
    "Politicians": {
        "SenSchumer": [0.893, 0.411, 0.948, 0.915, 0.972],
        "BernieSanders": [0.89, 0.402, 0.918, 0.776, 0.89],
        "SpeakerPelosi": [0.89, 0.429, 0.931, 0.884, 0.955],
        "GovRonDeSantis": [0.406, 0.23, 0.984, 0.872, 0.974],
        "ALL": [0.871, 0.413, 0.931, 0.856, 0.938],
    },
    "TV shows": {
        "60Minutes": [0.849, 0.433, 0.948, 0.897, 0.973],
        "etnow": [0.802, 0.59, 0.888, 0.617, 0.876],
        "nbcsnl": [0.869, 0.482, 0.946, 0.773, 0.916],
        "SportsCenter": [0.704, 0.25, 0.842, 0.52, 0.797],
        "ALL": [0.7924, 0.39025, 0.92645, 0.6811, 0.8733],
    },
    "Comedians": {
        "chrisrock": [0.858, 0.391, 0.864, 0.704, 0.917],
        "joerogan": [0.576, 0.193, 0.958, 0.708, 0.906],
        "SarahKSilverman": [0.899, 0.415, 0.949, 0.811, 0.944],
        "danieltosh": [0.795, 0.344, 0.931, 0.609, 0.847],
        "ALL": [0.838, 0.409, 0.933, 0.757, 0.919],
    },
    "News outlets": {
        "HuffPost": [0.869, 0.461, 0.92, 0.851, 0.955],
        "USATODAY": [0.785, 0.403, 0.923, 0.819, 0.945],
        "OANN": [0.351, 0.223, 0.985, 0.896, 0.978],
        "washingtonpost": [0.852, 0.397, 0.924, 0.857, 0.945],
        "ALL": [0.777, 0.363, 0.923, 0.815, 0.93],
    },
}
# Sheet holding each Figure 2 category
FIGURE2_SHEETS = {
    "Musical artists": "MusicalArtist",
    "Car makers": "cars",
    "Politicians": "Politician",
    "TV shows": "TelevisionShow",
    "Comedians": "Comedian",
    "News outlets": "news",
}
