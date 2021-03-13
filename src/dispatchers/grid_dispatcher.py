from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, GridSearchCV

import config


grids = {
	"halving": HalvingRandomSearchCV(
		None,
		config.GRID_SEARCH_PARAMS,
		n_jobs=-1,
		verbose=100
		),
    "grid": GridSearchCV(
		None,
		config.GRID_SEARCH_PARAMS,
		n_jobs=-1
  		),
}

