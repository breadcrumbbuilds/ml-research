import os
import sys
import json
import argparse
from sklearn import model_selection
from data import retrieve_data, retrieve_labels, read_labels_from_file

import config
import files



def run(binary_filename, name, label, force_rewrite=False):
    cwd = os.path.dirname(os.path.realpath(__file__))      #path to current file
    parent_dir = os.path.split(cwd)[0]
    binary_path = os.path.join(parent_dir, binary_filename)

    print('retrieving data')

    if label:
        data, dictionary = retrieve_data(binary_path)
        idx = dictionary['band names'].index(label)
        data = data[idx]
        print(data.shape)
    else:
        data, dictionary = retrieve_data(binary_path)



    dirs = config.BINARY_FILE.replace('raw', 'processed').split('/')
    new_parent_dir = parent_dir
    for dir in dirs[:-1]:
        new_parent_dir = os.path.join(new_parent_dir, dir)
        if not os.path.exists(new_parent_dir):
            print(f"+w {new_parent_dir}")
            os.mkdir(new_parent_dir)

    if label:
        csv_path = os.path.join(new_parent_dir, f"{label}.csv")
    else:
        csv_path = os.path.join(new_parent_dir, f"{name}.csv")

    if not label:

        json_path = csv_path.replace('csv', 'json')
        with open(json_path, 'w') as fp:
            print(f"+w {json_path}")
            json.dump(dictionary, fp, indent=4)

    if not os.path.exists(csv_path) or force_rewrite:
        print(f"+w {csv_path}")
        data.to_csv(csv_path, index=False)
    else:
        print(f"{csv_path} exists, no write")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--outputname", type=str)
    parser.add_argument("--target", type=str)
    parser.add_argument("--force", type=bool)


    args = parser.parse_args()

    run(config.BINARY_FILE,
        args.outputname,
        args.target,
        force_rewrite=args.force
        )