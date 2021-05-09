# src/train.py

import os
import time
import joblib
import argparse

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

import config
import files
from dispatchers import model_dispatcher, normalize_dispatcher, data_manip_dispatcher, grid_dispatcher

# label may need to be a param
def run(model, normalize=False, data_manip=None, grid=None):

    file = config.DATA_FILE
    df = pd.read_csv(config.DATA_FILE)
    df.fillna(0, inplace=True)
    for file in config.TRAINING_FILES:

        target = pd.read_csv(file)

        x_train, x_valid, y_train, y_valid = train_test_split(df.values, target.values, random_state=42)

        if normalize:
            scaler = normalize_dispatcher.normalize[normalize]
            x_train = scaler.fit_transform(x_train)
            x_valid = scaler.transform(x_valid)


        if data_manip:
            sampler = data_manip_dispatcher.data_manipulators[data_manip];
            x_train, y_train = sampler.fit_resample(x_train, y_train)


        vals, counts = np.unique(y_train, return_counts=True)


        print(f"X_train Shape: {x_train.shape}")
        print("Training Set Distribution")
        print(vals)
        print(counts)


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

        print(f"F1Score={f1_score}")


        model_fn = f"{files.extract_class_from_filename(file)}_{model}_{grid}_{data_manip}.bin"

        if grid:
            files.dump_model(clf.best_estimator_, model_fn)
            files.create_results_file(model_fn.replace('.bin', '.json'),
                                    {
                                        'model-file': model_fn,
                                        'training-file': file,
                                        'model-dispatch': model,
                                        'params': clf.best_estimator_.get_params(),
                                        'normalization-dispatch': normalize,
                                        'data-manip-dispatch': data_manip,
                                        'grid-dispatch': grid,
                                        'time-to-fit (s)': fit_time,
                                        'time-to-predict (s)': pred_time,
                                        'f1': f1_score,
                                    })
        else:
            files.dump_model(clf, model_fn)
            files.create_results_file(model_fn.replace('.bin', '.json'),
                                    {
                                        'model-file': model_fn,
                                        'training-file': file,
                                        'model-dispatch': model,
                                        'params': clf.get_params(),
                                        'normalization-dispatch': normalize,
                                        'data-manip-dispatch': data_manip,
                                        'grid-dispatch': grid,
                                        'time-to-fit (s)': fit_time,
                                        'time-to-predict (s)': pred_time,
                                        'f1': f1_score,
                                    })

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--label",
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

    run(
        model=args.model,
        normalize=args.normalize,
        data_manip=args.datamanip,
        grid=args.grid)