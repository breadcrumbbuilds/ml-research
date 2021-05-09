import os
import numpy as np
import pandas as pd
import json


def read_header(binary_filename):
    '''
		reads the header file for the binary file name
		passed. returns a dictionary of the file contents
    '''
    header_filename = binary_filename.replace('.bin', '.hdr')

    if not os.path.exists(header_filename):
    	raise Exception(f"{header_filename} does not exist")

    continue_reading = False
    result = {}
    dict_key = None
    keys = ["samples", "lines", "bands", "band names"]

    print(f"+r {header_filename}")
    with open(header_filename) as f:
        lines = f.readlines()
        for line in lines:

            line_split = line.split("=")
            line_split = [l.strip().replace(',', '') for l in line_split]

            if len(line_split) < 2:
                if not continue_reading:
                    continue

                if "}" in value:
                    continue_reading = False
                    result[key].append(value.replace('}', ''))
                    continue
                result[key].append(line_split[0])
                continue
            key = line_split[0]
            value = line_split[1]

            if key in keys:
                if "{" in value:
                    continue_reading = True
                    result[key] = list()
                    result[key].append(value.replace('{', '').replace(',','').replace('}', ''))
                    continue


                result[key] = value
    return result


def read_binary(binary_path):
    dictionary = read_header(binary_path)
    print(f"+r {binary_path}")
    data = np.fromfile(binary_path, '<f4')
    return data, dictionary


def bsq_to_scikit(ncol, nrow, nband, data):
    print("Converting bsq to Sklearn Format")
    npx = nrow * ncol # number of pixels
    print(f"Columns={ncol}, Rows={nrow}, Bands={nband}")
    # convert the image data to a numpy array of format expected by sgd
    img_np = np.zeros((npx, nband))

    for i in range(0, nrow):
        ii = i * ncol
        for j in range(0, ncol):
            for k in range(0, nband):
                # don't mess up the indexing
                img_np[ii + j, k] = data[(k * npx) + ii + j]
    print(img_np.shape)
    return img_np


def retrieve_data(binary_path):
    csv_path = binary_path.replace('bin', 'csv')

    if not os.path.exists(binary_path):
        raise Exception(f"{binary_path} doesn't exists")

    print("creating csv from binary file")

    data, dictionary = read_binary(binary_path)
    data = bsq_to_scikit(int(dictionary['samples']),
                int(dictionary['lines']),
                int(dictionary['bands']),
                data)
    data = pd.DataFrame(data)
    data.to_csv(csv_path)

    return data, dictionary


def convert_y_to_binary(target, y):
    print(f'converting {target} to binary')
    ones = np.ones((y.shape))
    vals = np.sort(np.unique(y))
    # create an array populate with the false value
    t = ones * vals[len(vals) - 1]
    if target.lower() == 'water':
        y = np.not_equal(y, t).astype(int)
        vals, counts = np.unique(y, return_counts=True)
        print(vals, counts)
    else:
        y = np.logical_and(y, t).astype(int)
        vals, counts = np.unique(y, return_counts=True)
        print(vals, counts)

    return y


def retrieve_labels(path, label):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    initialized = False
    for file in files:
        filepath = os.path.join(path, file)
        if '.bin' in file:
            if label.lower() in file.lower():
                data, dictionary = read_binary(filepath)
                target = "target"
                dictionary['band names'] = target
                if "full" or "zoom" in filepath:
                    data = convert_y_to_binary(file.replace('.bin', ''), data)
                if not initialized:
                    df = pd.DataFrame(data, columns=[target])
                    initialized = True
    return df

def read_labels_from_file(path, label):
    data, dictionary = read_binary(path)
    data = bsq_to_scikit(int(dictionary['samples']),
                int(dictionary['lines']),
                int(dictionary['bands']),
                data)
    print(dictionary['band names'])
    print(data.shape)
    df = pd.DataFrame(data, columns=[x.replace(',','') for x in dictionary['band names']])

    return df

