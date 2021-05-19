import os
import sys
import json
import argparse
from sklearn import model_selection
from data import retrieve_data, retrieve_labels, read_labels_from_file

import config
import files



def run(folds, output_file_name, force_rewrite=False):
    cwd = os.path.dirname(os.path.realpath(__file__))      #path to current file
    parent_dir = os.path.split(cwd)[0]
    binary_data_path = os.path.join(parent_dir, config.BINARY_DATA_FILE)
    binary_label_path = os.path.join(parent_dir, config.BINARY_MASK_FILE)

    print('retrieving labels')
    label, l_dict = retrieve_data(binary_label_path)

    print('retrieving data')
    data, d_dict = retrieve_data(binary_data_path)
    data["target"] = -1


    result = None

    print('gathering labelled examples')
    for column in label.columns:
        indices = label.index[label[column] == 1].tolist()

        if result is None:
            data.loc[indices, "target"] = column
            result = data.loc[indices]
        else:
            data.loc[indices, "target"] = column
            result = result.append(data.loc[indices])

    result["kfold"] = -1

    result = result.sample(frac=1).reset_index(drop=True)

    print('splitting into folds')

    y = result.target.values
    kf = model_selection.StratifiedKFold(n_splits=folds)

    for f, (t_, v_) in enumerate(kf.split(X=result, y=y)):
        result.loc[v_, 'kfold'] = f

    dirs = config.BINARY_FILE.replace('raw', 'processed').split('/')
    new_parent_dir = parent_dir
    for dir in dirs[:-1]:
        new_parent_dir = os.path.join(new_parent_dir, dir)
        if not os.path.exists(new_parent_dir):
            print(f"+w {new_parent_dir}")
            os.mkdir(new_parent_dir)


    csv_path = os.path.join(new_parent_dir, f"{output_file_name}.csv")

    if not os.path.exists(csv_path) or force_rewrite:
        print(f"+w {csv_path}")
        result.to_csv(csv_path, index=False)
    else:
        print(f"{csv_path} exists, no write")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--folds", type=int)
    parser.add_argument("--outputname", type=str)
    parser.add_argument("--force", type=bool)


    args = parser.parse_args()

    run(
        args.folds,
        args.outputname,
        force_rewrite=args.force
        )