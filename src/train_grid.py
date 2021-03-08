# src/train.py
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import config
import create_folds
import model_dispatcher

# label may need to be a param
def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop("target", axis=1).values
    y_train = df_train.target.values

    x_valid = df_valid.drop("target", axis=1).values
    y_valid = df_valid.target.values

    clf = GridSearchCV(model_dispatcher.models[model],
                       config.GRID_SEARCH_PARAMS,
                       n_jobs=-1,
                       scoring='f1'
                       )

    clf.fit(x_train, y_train)

    preds = clf.best_estimator_.predict(x_valid)

    accuracy = metrics.f1_score(y_valid, preds)

    print(f"Fold={fold}, Accuracy={accuracy}")
    print(f'Best Params: {clf.best_params_}')

    joblib.dump(clf,
                os.path.join(config.MODEL_OUTPUT, f"dt_{model}_{fold}.bin")
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )

    args = parser.parse_args()

    run(fold=args.fold, model=args.model)