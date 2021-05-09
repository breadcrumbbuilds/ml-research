from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline

import config


data_manipulators = {
	"random-us": RandomUnderSampler(
            sampling_strategy="majority"
		),

    "random-us-.5": RandomUnderSampler(
            sampling_strategy=0.5
		),

    "random-os":RandomOverSampler(
        sampling_strategy='minority',
        ),

    "random-os-.5":RandomOverSampler(
        sampling_strategy=0.5,
        ),

    "smote": SMOTE(),

    "smote-.05": SMOTE(
        sampling_strategy=0.05
        ),

    "smote-.1": SMOTE(
        sampling_strategy=0.1
        ),

    "smote-.2": SMOTE(
        sampling_strategy=0.2
        ),

    "svmsmote": SVMSMOTE(),

    "bl-smote": BorderlineSMOTE(),

    "smote-pipeline": Pipeline(
        steps=[
            ('over', SMOTE(sampling_strategy=0.00025)),
             ('under', RandomUnderSampler(sampling_strategy=.25))
        ])
}