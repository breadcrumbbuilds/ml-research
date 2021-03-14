# src/files.py
import os
import joblib

import config

def dump_model(clf, filename):
    ''' standardize saving of model '''
    path = os.path.join(config.MODEL_OUTPUT, filename)

    create_path(path)

    print(f'+w {path}')
    joblib.dump(clf,
                path)


def create_path(path):

    path_splits = path.split('/')
    results = os.curdir

    for path in path_splits[:-1]:

        results += "/" + path

        if not os.path.exists(results):
            os.mkdir(results)


def extract_class_from_filename(fn):
    splits = fn.split('.')[0]
    splits = splits.split('/')

    filename = splits[len(splits)-1]
    file_name = filename.split('_')
    return file_name[len(file_name) - 1]


def get_bin_path():
    cwd = os.path.dirname(os.path.realpath(__file__))      #path to current file
    parent_dir = os.path.split(cwd)[0]
    binary_path = os.path.join(parent_dir, config.BINARY_FILE)

    return binary_path
