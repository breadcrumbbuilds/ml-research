#!/bin/bash

python src/create_folds.py --folds 5 --label water --force True
python src/create_folds.py --folds 5 --label broadleaf --force True
python src/create_folds.py --folds 5 --label conifer --force True
python src/create_folds.py --folds 5 --label herb --force True
python src/create_folds.py --folds 5 --label mixed --force True
python src/create_folds.py --folds 5 --label ccutbl --force True
python src/create_folds.py --folds 5 --label exposed --force True
python src/create_folds.py --folds 5 --label rivers --force True
python src/create_folds.py --folds 5 --label shrub --force True
