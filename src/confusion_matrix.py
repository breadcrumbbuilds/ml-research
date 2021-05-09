# Taken directly from sklearn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import config
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

def run(label):
    file = config.DATA_FILE

    df = pd.read_csv(config.DATA_FILE)
    df.fillna(0, inplace=True)
    target = pd.read_csv(config.TARGET_FILE)
    clf = joblib.load(config.INFERENCE_MODEL)

    x_train, x_valid, y_train, y_valid = train_test_split(df.values, target.values, random_state=42)


    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    titles_options = [(f"Confusion matrix, without normalization: {label}", None),
                    (f"Normalized confusion matrix: {label}", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, x_valid, y_valid,
                                    display_labels=["false", "true"],
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)
        plt.suptitle(config.INFERENCE_MODEL)

        print(title)
        print(disp.confusion_matrix)

        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
         "--label", type=str
    )

    args = parser.parse_args()

    run(label=args.label)