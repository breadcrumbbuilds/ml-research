from sklearn.experimental import enable_halving_search_cv
from sklearn.preprocessing import Normalizer, StandardScaler

import config


normalize = {
	"normalize_l1": Normalizer(norm="l1"),
    "normalize_l2": Normalizer(norm="l2"),
    "normalize_max":Normalizer(norm="max"),
    "standardize_wo_std": StandardScaler(with_std=False),
    "standardize_wo_mean": StandardScaler(with_mean=False),
    "standardize": StandardScaler()
}

