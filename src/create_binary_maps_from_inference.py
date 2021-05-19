import pickle
import numpy
import config
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im

labels = [
    'blowdownfireweed',
    'deciduous',
    'fireweedgrass',
    'lake',
    'pineburnedfireweed',
    'blowdownlichen',
    'exposed',
    'grass',
    'pineburned',
    'windthrowgreenherbs'
    ]

with open(config.INFERENCE_FILE, 'rb') as f:
    data = pickle.load(f)
    vals, counts = np.unique(data, return_counts=True)
    for idx, val in enumerate(vals):
        binary = data == val
        fn = f'output/{labels[idx]}_pred.bin'
        with open(fn, 'wb') as fn:
            pickle.dump(binary, fn)

        plt.axis('off')
        plt.imshow(binary, cmap='gray')

        plt.savefig(f'output/{labels[idx]}_pred.png', transparent=True, bbox_inches='tight', pad_inches=0)