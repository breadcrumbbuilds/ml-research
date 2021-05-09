from sklearn import tree
from sklearn import svm
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier

models = {
	"decision-tree-gini": tree.DecisionTreeClassifier(
		criterion="gini",
		max_depth=5
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
     n_estimators=10000,
     max_depth=3,
     n_jobs=-1
     ),
 	"rf-stump": ensemble.RandomForestClassifier(
     n_estimators=1000,
     max_depth=1,
     n_jobs=-1
     ),
	"svm": svm.SVC(cache_size=1500),
	"svm-opt-water": svm.SVC(C=11, degree=3, kernel='rbf', probability=True),

	"adaboost": ensemble.AdaBoostClassifier(),
 	"gradientboost": ensemble.GradientBoostingClassifier(),
	"extratrees": ensemble.ExtraTreesClassifier(),
	"vote-ensemble": ensemble.VotingClassifier(n_jobs=-1, estimators=[("rf",ensemble.RandomForestClassifier()),
                                                          ("et",ensemble.ExtraTreesClassifier()),
                                                          ("ab",ensemble.AdaBoostClassifier()),
                                                          ("gb", ensemble.GradientBoostingClassifier())]),

	"mlp": MLPClassifier(max_iter=1000, verbose=10),
}