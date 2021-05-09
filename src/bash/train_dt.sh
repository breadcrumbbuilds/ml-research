#!/bin/bash

python src/train.py --model decision-tree-gini --normalize standardize --datamanip smote-.05 --fold 0
python src/train.py --model decision-tree-gini --normalize standardize --datamanip smote-.05 --fold 1
python src/train.py --model decision-tree-gini --normalize standardize --datamanip smote-.05 --fold 2
python src/train.py --model decision-tree-gini --normalize standardize --datamanip smote-.05 --fold 3
python src/train.py --model decision-tree-gini --normalize standardize --datamanip smote-.05 --fold 4

