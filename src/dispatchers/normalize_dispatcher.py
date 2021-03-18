from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import Normalizer, StandardScaler

import config


normalize = {
	"normalize-l1": Normalizer(norm="l1"),
    "normalize-l2": Normalizer(norm="l2"),
    "normalize-max":Normalizer(norm="max"),
    "standardize-wo-std": StandardScaler(with_std=False),
    "standardize-wo-mean": StandardScaler(with_mean=False),
    "standardize": StandardScaler()
}

