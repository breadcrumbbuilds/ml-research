import os
import json
import joblib
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import files
import config
from data import retrieve_data
from dispatchers import normalize_dispatcher

def run(predict, normalize):
    if not predict:
        predict = 'class'
    binary_path = files.get_bin_path()

    data, dictionary = retrieve_data(binary_path)

    cols, rows, bands = int(dictionary['lines']), int(dictionary['samples']), int(dictionary['bands'])

    x = data.values.reshape((cols * rows, bands))

    if normalize:
        scaler = normalize_dispatcher.normalize[normalize]
        print(f'normalizing data using {scaler}')
        x = scaler.fit_transform(x)

    clf = joblib.load(config.INFERENCE_MODEL)


    if predict == 'class':
        print('predicting class')
        prediction = clf.predict(x)
    elif predict == 'proba':
        print('predicting probability')
        prediction = clf.predict_proba(x)[:,1]
    else:
        raise Exception("Not a recognized prediction type")


    prediction = prediction.reshape((cols, rows))

    plt.imshow(prediction, cmap='gray')
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predict",
        type=str
    )
    parser.add_argument(
        "--normalize",
        type=str
    )


    args = parser.parse_args()
    run(predict=args.predict, normalize=args.normalize)