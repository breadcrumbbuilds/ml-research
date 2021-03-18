#!/bin/bash

python src/train.py --model rf --normalize standardize --datamanip smote-.1 --fold 0 --grid grid
python src/train.py --model rf --normalize standardize --datamanip smote-.1 --fold 1 --grid grid
python src/train.py --model rf --normalize standardize --datamanip smote-.1 --fold 2 --grid grid
python src/train.py --model rf --normalize standardize --datamanip smote-.1 --fold 3 --grid grid
python src/train.py --model rf --normalize standardize --datamanip smote-.1 --fold 4 --grid grid

