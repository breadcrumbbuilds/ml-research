# src/train.py

import os
import time
import joblib
import argparse

import numpy as np
import pandas as pd
from sklearn import metrics

import config
import files
from dispatchers import model_dispatcher, normalize_dispatcher, data_manip_dispatcher, grid_dispatcher

# label may need to be a param
def run(fold, model, normalize=False, data_manip=None, grid=None):
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


    if data_manip:
        sampler = data_manip_dispatcher.data_manipulators[data_manip];
        x_train, y_train = sampler.fit_resample(x_train, y_train)

    print(x_train.shape)
    vals, counts = np.unique(y_train, return_counts=True)

    if grid:
        clf = grid_dispatcher.grids[grid]
        clf.estimator = model_dispatcher.models[model]
        clf.scoring = 'f1'
    else:
        clf = model_dispatcher.models[model]

    start_fit = time.time()

    # Fitting
    clf.fit(x_train, y_train)

    end_fit = time.time()

    fit_time = end_fit - start_fit


    start_pred = time.time()

    # Predicting
    preds = clf.predict(x_valid)

    end_pred = time.time()

    pred_time = end_pred - start_pred

    f1_score = metrics.f1_score(y_valid, preds)

    print(f"Fold={fold}, F1Score={f1_score}")

    fn = f"{files.extract_class_from_filename(config.TRAINING_FILE)}_{model}_{grid}_{data_manip}_{fold}.bin"
    files.dump_model(clf, fn)
    if grid:
        files.create_results_file(fn.replace('.bin', '.json'),
                                {
                                    'model-file': fn,
                                    'training-file': config.TRAINING_FILE,
                                    'model-dispatch': model,
                                    'params': clf.best_estimator_.get_params(),
                                    'normalization-dispatch': normalize,
                                    'data-manip-dispatch': data_manip,
                                    'grid-dispatch': grid,
                                    'time-to-fit (s)': fit_time,
                                    'time-to-predict (s)': pred_time,
                                    'fold': fold,
                                    'f1': f1_score,
                                })
    else:
        files.create_results_file(fn.replace('.bin', '.json'),
                                {
                                    'model-file': fn,
                                    'training-file': config.TRAINING_FILE,
                                    'model-dispatch': model,
                                    'params': clf.get_params(),
                                    'normalization-dispatch': normalize,
                                    'data-manip-dispatch': data_manip,
                                    'grid-dispatch': grid,
                                    'time-to-fit (s)': fit_time,
                                    'time-to-predict (s)': pred_time,
                                    'fold': fold,
                                    'f1': f1_score,
                                })

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
    parser.add_argument(
        "--datamanip",
        type=str
    ),
    parser.add_argument(
        "--grid",
        type=str
    )

    args = parser.parse_args()

    run(fold=args.fold,
        model=args.model,
        normalize=args.normalize,
        data_manip=args.datamanip,
        grid=args.grid)