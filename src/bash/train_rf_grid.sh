#!/bin/bash

python src/train.py --model rf --normalize standardize --datamanip smote-.05 --fold 0 --grid grid
python src/train.py --model rf --normalize standardize --datamanip smote-.05 --fold 1 --grid grid
python src/train.py --model rf --normalize standardize --datamanip smote-.05 --fold 2 --grid grid
python src/train.py --model rf --normalize standardize --datamanip smote-.05 --fold 3 --grid grid
python src/train.py --model rf --normalize standardize --datamanip smote-.05 --fold 4 --grid grid

