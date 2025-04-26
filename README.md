# SensorDBSCAN: Semi-Supervised Learning for Fault Diagnosis in Chemical Processes

This repository is the modified official implementation of methods from the paper [SensorSCAN: Self-Supervised Learning and Deep Clustering for Fault Diagnosis in Chemical Processes](https://www.sciencedirect.com/science/article/abs/pii/S0004370223001583).

## Our modifications

* Changed second learning stage from Deep Clustering to semi-supervised encoder training with triplet loss (to make clusters separable) with DBSCAN applied later (so we don't have to specify number of clusters and could use margin from triplet loss to estimate epsilon for DBSCAN)
* Added active learning for semi-supervised learning (data sampling for labeling and triplets sampling for training)
* Added intermediate clusteing quality evaluation mechanism to estimate moment when we should sample more data

## Requirements

To install dependencies, run 
```
pip install -r requirements.txt
```

## Running experiments

Running experiments involves training and evaluating processes defined for each model in [models](/models/) folder. To run experiments, use the command line interface `python main.py` with the argument `--config-name {model}_{dataset}`. For example:

```
python main.py --config-name pca_kmeans_rieth_tep
```

Pretrained models are stored in [saved_models](/saved_models/) folder. To evaluate a pretrained model, use `path_to_model` argument:

```
python main.py --config-name pca_kmeans_rieth_tep path_to_model=saved_models/pca_kmeans_rieth_tep.joblib
```

Experimental results are stored in [results](/results/) folder. All arguments are defined in [configs](/configs/) folder.

## Experimental results

Results on `rieth_tep`

|                      | ACC     | ARI    | NMI    | Detection TPR | Detection FPR | CDR    | ADD    |
|----------------------|---------|--------|--------|---------------|---------------|--------|--------|
| `pca_kmeans`         | 0.2745  | 0.1100 | 0.3634 | 0.3590        | 0.0000        | 0.7910 | 113.95 |
| `st_catgan`          | 0.1754  | 0.1135 | 0.2223 | 0.3044        | 0.0000        | 0.3238 | 102.63 |
| `convae`             | 0.1794  | 0.1565 | 0.2537 | 0.3631        | 0.0008        | 0.3664 | 164.76 |
| `sensorscan`         | 0.5926  | 0.4747 | 0.6812 | 0.7316        | 0.0014        | 0.7351 | 57.15  |
| `sensorscan (paper)` | 0.7850  | 0.7030 | 0.8460 | 0.8400        | 0.0002        | 0.9200 | 5.21   |
| `sensordbscan`       | 0.7387  | 0.7124 | 0.8031 | 0.6877        | 0.0000        | 0.6913 | 121.23 |

Results on `reinartz_tep`

|                      | ACC    | ARI    | NMI    | Detection TPR | Detection FPR | CDR    | ADD    |
|----------------------|--------|--------|--------|---------------|---------------|--------|--------|
| `pca_kmeans`         | 0.3513 | 0.1316 | 0.4484 | 0.3581        | 0.0000        | 0.9562 | 113.33 |
| `st_catgan`          | 0.3016 | 0.1287 | 0.3606 | 0.3627        | 0.0001        | 0.8882 | 135.04 |
| `convae`             | 0.4975 | 0.2381 | 0.5863 | 0.6023        | 0.0016        | 0.9402 | 155.16 |
| `sensorscan`         | 0.5287 | 0.3336 | 0.7551 | 0.9013        | 0.0002        | 0.7219 | 30.98  |
| `sensorscan (paper)` | 0.7360 | 0.4810 | 0.8500 | 0.8700        | 0.0002        | 0.9600 | 28.47  |
| `sensordbscan`       | 0.7808 | 0.5412 | 0.8154 | 0.7808        | 0.0000        | 0.8433 | 165.35 |
