import os
import sys
import json
import argparse
from sklearn import model_selection
from data import retrieve_data, retrieve_labels, read_labels_from_file

import config
import files



def run(binary_filename, label, label_is_file, folds=5, force_rewrite=False):
    cwd = os.path.dirname(os.path.realpath(__file__))      #path to current file
    parent_dir = os.path.split(cwd)[0]
    binary_path = os.path.join(parent_dir, binary_filename)
    labels_path = os.path.join(parent_dir, config.LABEL_DIR)

    print('retrieving data')
    data, dictionary = retrieve_data(binary_path)

    print('retrieving label')
    if label_is_file:
        labels = read_labels_from_file(config.LABEL_FILE, label)

    else:
        labels = retrieve_labels(labels_path, label)
    dataset = data.join(labels)

    print(dataset.head())
    print(dataset.describe())

    print('creating KFolds')
    dataset["kfold"] = -1

    dataset = dataset.sample(frac=1).reset_index(drop=True)

    y = dataset.target.values
    kf = model_selection.StratifiedKFold(n_splits=folds)

    for f, (t_, v_) in enumerate(kf.split(X=dataset, y=y)):
        dataset.loc[v_, 'kfold'] = f


    dirs = config.BINARY_FILE.replace('raw', 'processed').split('/')
    new_parent_dir = parent_dir
    for dir in dirs[:-1]:
        new_parent_dir = os.path.join(new_parent_dir, dir)
        if not os.path.exists(new_parent_dir):
            print(f"+w {new_parent_dir}")
            os.mkdir(new_parent_dir)
    if "S2A" in binary_path:
        csv_path = os.path.join(new_parent_dir, f"s2a_train_folds_{label}.csv").replace('raw', 'processed')
    elif "L8" in binary_path:
        csv_path = os.path.join(new_parent_dir, f"l8_train_folds_{label}.csv").replace('raw', 'processed')

    json_path = csv_path.replace('csv', 'json')
    with open(json_path, 'w') as fp:
        print(f"+w {json_path}")
        json.dump(dictionary, fp, indent=4)

    if not os.path.exists(csv_path) or force_rewrite:
        print(f"+w {csv_path}")
        dataset.to_csv(csv_path, index=False)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--label", type=str)
    parser.add_argument("--labelfile", type=bool)
    parser.add_argument("--force", type=bool)
    parser.add_argument("--folds", type=int)


    args = parser.parse_args()

    run(config.BINARY_FILE,
        args.label,
        args.labelfile,
        folds=args.folds,
        force_rewrite=args.force
        )