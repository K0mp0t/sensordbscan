# SensorDBSCAN: Semi-Supervised Learning for Fault Diagnosis in Chemical Processes

This repository is based on code from [SensorSCAN: Self-Supervised Learning and Deep Clustering for Fault Diagnosis in Chemical Processes](https://www.sciencedirect.com/science/article/abs/pii/S0004370223001583).
We've been accepted to IEEE Access with this method: [Paper](https://ieeexplore.ieee.org/document/10869347)

## Our modifications

* Changed second learning stage from Deep Clustering to semi-supervised encoder training with triplet loss 
(to make clusters separable) with DBSCAN applied later (so we don't have to specify number of clusters and could use 
triplet loss margin hyperparameter as epsilon for DBSCAN)
* Added active learning for semi-supervised learning (data sampling for labeling and effective triplets ranking for training). Made it effective: ~700 labeled samples is enough for a decent performance on a 2M dataset
* Added intermediate clusteing quality evaluation mechanism to estimate moment when we should sample more data
* Changed fixed positional embeddings (as is origianl Transformer paper) to RoPE embeddings
* Developed outlier handling technique (density-based clustering algorithms may produce `outlier` cluster)
* (Wasn't included in the IEEE Access paper) Developed a more complex and therefore more accurate sequence pooling mechanism

## Requirements

To install dependencies, run 
```
conda create -n sensordbscan --file environment.yml
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

**Metrics in this section is fixed to date of IEEE Access paper submission. Some further experiments and improvements made our algorithm achieve better results** 

Results on `rieth_tep`

|| ACC    | ARI    | NMI    | Detection TPR | Detection FPR | CDR    | ADD    |
|-|--------|--------|--------|---------------|---------------|--------|--------|
|`PCA+KMeans`| 0.2745 | 0.1100 | 0.3634 | 0.3590        | 0.0000        | 0.7910 | 113.95 |
|`ST-CATGAN`| 0.1754 | 0.1135 | 0.2223 | 0.3044        | 0.0000        | 0.3238 | 102.63 |
|`ConvAE`| 0.1794 | 0.1565 | 0.2537 | 0.3631        | 0.0008        | 0.3664 | 164.76 |
|`SensorSCAN`| 0.5926 | 0.4747 | 0.6812 | 0.7316        | 0.0014        | 0.7351 | 57.15  |
|`SensorSCAN (paper)`| 0.7850 | 0.7030 | 0.8460 | 0.8400        | 0.0002        | 0.9200 | 5.21   |
|`SensorDBSCAN`| 0.7393 | 0.7124 | 0.8029 | 0.6877 | 0.0000        | 0.6913 | 121.42 |

Results on `reinartz_tep`

|                        | ACC    | ARI    | NMI    | Detection TPR | Detection FPR | CDR    | ADD    |
|------------------------|--------|--------|--------|---------------|--------|--------|--------|
| `PCA+KMeans`           | 0.3513 | 0.1316 | 0.4484 | 0.3581        |0.0000| 0.9562 | 113.33 |
| `ST-CATGAN`            | 0.3016 | 0.1287 | 0.3606 | 0.3627        |0.0001| 0.8882 | 135.04 |
| `ConvAE`               | 0.4975 | 0.2381 | 0.5863 | 0.6023        |0.0016| 0.9402 | 155.16 |
| `SensorSCAN`           | 0.5287 | 0.3336 | 0.7551 | 0.9013        |0.0002| 0.7219 | 30.98  |
| `SensorSCAN (paper)`   | 0.7360 | 0.4810 | 0.8500 | 0.8700        |0.0002| 0.9600 | 28.47  |
| `SensorDBSCAN (paper)` | 0.7379 | 0.5410 | 0.8154 | 0.7808        |0.0000| 0.8433 | 164.91 |
| `SensorDBSCAN (best)`  | 0.8297 | 0.6706 | 0.8800 | 0.8517        |0.0009| 0.9064 | 36.59  |
