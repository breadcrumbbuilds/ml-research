# Machine Learning Research
This repo contains the continuation of a proof of concept for a Machine Learning directed study course. The focus of the project is predicting fuel-type layers from satellite images. I've recently worked on automating some of the scripting using `Approaching Almost Any Machine Learning Problem` as a reference. 

## Getting Started
1. You'll need the data... which I'm not at liberty to distribute (sorry)
2. Next, you'll need to take that raw data and generate some training files in the form of csv
(Description of config.py)
```python
python src/create_folds.py --label <the label you want to create a model for> --folds <number of k folds (stratified)>
```
