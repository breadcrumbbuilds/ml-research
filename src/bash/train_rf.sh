#!/bin/bash

python src/train.py --model rf-shallow --normalize standardize --datamanip smote-.1 --fold 0
python src/train.py --model rf-shallow --normalize standardize --datamanip smote-.1 --fold 1
python src/train.py --model rf-shallow --normalize standardize --datamanip smote-.1 --fold 2
python src/train.py --model rf-shallow --normalize standardize --datamanip smote-.1 --fold 3
python src/train.py --model rf-shallow --normalize standardize --datamanip smote-.1 --fold 4
