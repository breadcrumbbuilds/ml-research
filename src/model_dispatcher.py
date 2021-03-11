from sklearn import tree
from sklearn import ensemble

models = {
	"decision_tree_gini": tree.DecisionTreeClassifier(
		criterion="gini"
	),
 	"decision_tree_entropy": tree.DecisionTreeClassifier(
      criterion="entropy",

	),
	"rf": ensemble.RandomForestClassifier(),
	"rf_after_grid": ensemble.RandomForestClassifier(
		criterion='gini',
		max_depth=9,
		max_features='auto',
		n_estimators=200,
		n_jobs=14
	),
	"rf_shallow": ensemble.RandomForestClassifier(
     max_depth=3,
     n_jobs=-1
     ),
}