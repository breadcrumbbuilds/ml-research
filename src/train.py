# src/train.py
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics

import config
import files
from dispatchers import model_dispatcher, normalize_dispatcher

# label may need to be a param
def run(fold, model, normalize=False):
    df = pd.read_csv(config.TRAINING_FILE)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    df_train = df_train.drop("kfold", axis=1)
    df_valid = df_valid.drop("kfold", axis=1)


    x_train = df_train.drop("target", axis=1).values
    y_train = df_train.target.values

    x_valid = df_valid.drop("target", axis=1).values
    y_valid = df_valid.target.values

    if normalize:
        scaler = normalize_dispatcher.normalize[normalize]
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)

    print(x_train.shape)

    clf = model_dispatcher.models[model]

    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.f1_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    fn = f"{files.extract_class_from_filename(config.TRAINING_FILE)}_{model}_{fold}.bin"
    files.dump_model(clf, fn)


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
    parser.add_argument(
        "--normalize",
        type=str
    )

    args = parser.parse_args()

    run(fold=args.fold,
        model=args.model,
        normalize=args.normalize)