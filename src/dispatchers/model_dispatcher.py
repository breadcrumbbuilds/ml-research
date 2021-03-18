from sklearn import tree
from sklearn import svm
from sklearn import ensemble

models = {
	"decision-tree-gini": tree.DecisionTreeClassifier(
		criterion="gini"
	),
 	"decision-tree-entropy": tree.DecisionTreeClassifier(
      criterion="entropy",

	),
	"rf": ensemble.RandomForestClassifier(n_jobs=-1),
	"rf-opt-water": ensemble.RandomForestClassifier(
		criterion='gini',
		max_depth=9,
		max_features='auto',
		n_estimators=200,
		n_jobs=14
	),
	"rf-shallow": ensemble.RandomForestClassifier(
     max_depth=3,
     n_jobs=-1
     ),
	"svm": svm.SVC(cache_size=1500),
	"svm-opt-water": svm.SVC(C=11, degree=3, kernel='rbf', probability=True),
}