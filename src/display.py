import matplotlib.pyplot as plt
import argparse
import pickle

import config

def run():
    with open(config.INFERENCE_FILE, 'rb') as f:
        data = pickle.load(f)
        plt.axis('off')
        plt.imshow(data)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    run()