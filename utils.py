from typing import Union, List

import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
from fddbenchmark import FDDDataset


def weighted_max_occurence(true_labels, cluster_labels, n_types):
    _cluster_labels = cluster_labels.values
    mapping = np.zeros(n_types, dtype='int')

    # for cluster_label in np.unique(_cluster_labels):
    #     cluster_Y = true_labels[_cluster_labels == cluster_label]
    #     unique_items, unique_counts = np.unique(cluster_Y, return_counts=True)
    #
    #     normal_type = np.where(unique_items == 0)[0]
    #     if normal_type.size == 1 and (unique_counts[normal_type[0]] / unique_counts.sum()) > (1 / (unique_items.size + 1)):
    #         mapping[cluster_label] = 0
    #     else:
    #         mapping[cluster_label] = np.random.choice(unique_items[unique_counts == unique_counts.max()])

    for type_idx in range(n_types):
        type_clusters = _cluster_labels[true_labels == type_idx]
        values, counts = np.unique(type_clusters, return_counts=True)
        mapping[type_idx] = values[np.argmax(counts)]

    return mapping


def build_costs_matrix(true_labels, cluster_labels, nclusters=None):
    assert len(true_labels) == len(cluster_labels)

    if nclusters is None:
        nclusters = np.unique(cluster_labels).shape[0]

    costs_matrix = np.zeros((max(true_labels)+1, nclusters), dtype=int)

    for label in np.unique(true_labels):
        type_clusters = cluster_labels[true_labels == label]
        cluster_labels_, counts = np.unique(type_clusters, return_counts=True)
        # counts = counts[cluster_labels_ >= 0]
        # cluster_labels_ = cluster_labels_[cluster_labels_ >= 0]
        costs_matrix[label, cluster_labels_] = counts

    return costs_matrix


def label_assignment(true_labels, cluster_labels):
    _cluster_labels = cluster_labels.values
    costs_matrix = build_costs_matrix(_cluster_labels, true_labels)

    n_types = np.unique(cluster_labels).shape[0]

    # noinspection PyArgumentList
    row_ind, col_ind = linear_sum_assignment(costs_matrix, maximize=True)
    mapping = np.zeros(n_types, dtype='int')
    mapping[row_ind] = col_ind

    for type_idx in range(n_types):
        if type_idx in row_ind:
            continue
        type_labels = true_labels[_cluster_labels == type_idx]
        unique_items, unique_counts = np.unique(type_labels, return_counts=True)
        mapping[type_idx] = unique_items[np.argmax(unique_counts)]

    if 0 not in mapping:
        mapping[costs_matrix[:, 0].argmax()] = 0

    return mapping


def exclude_columns(df):
    excl = [
        'xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26', 'xmeas_27', 
        'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32', 
        'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 
        'xmeas_38', 'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_5', 
        'xmv_9', 'xmeas_19'
    ]
    columns = [c for c in df.columns if c not in excl]
    return df[columns]


def exclude_classes(dataset: FDDDataset, classes_to_keep: Union[str, List[str]]):
    if classes_to_keep == 'all':
        return dataset

    dataset.train_mask = dataset.train_mask & dataset.label.isin(classes_to_keep)
    dataset.test_mask = dataset.test_mask & dataset.label.isin(classes_to_keep)


def normalize(dataset):
    mean = dataset.df[dataset.train_mask].mean(axis = 0)
    std = dataset.df[dataset.train_mask].std(axis = 0)
    dataset.df = (dataset.df - mean) / std

def make_rieth_imbalance(train_mask):
    # make rieth_tep imbalance: 500 normal runs and 5 runs for each fault
    train_runs = train_mask[train_mask].index.get_level_values(0).unique()
    imbalance_train_runs = train_runs[:600]
    imbalance_train_mask = train_mask.copy()
    imbalance_train_mask.loc[:] = False
    imbalance_train_mask.loc[imbalance_train_runs] = True
    return imbalance_train_mask

def take_dataset_fraction(dataset, cfg):
    labels = pd.merge(dataset.train_mask, dataset.label, on=['run_id', 'sample'])
    labels = labels[labels.train_mask]
    labels = labels.groupby('labels').sample(frac=cfg.fraction, random_state=cfg.random_seed)

    fractioned_train_mask = dataset.train_mask.copy()
    fractioned_train_mask.loc[:] = False
    fractioned_train_mask.loc[labels.index] = True

    return fractioned_train_mask


def print_clustering(metrics, logging):
    lines = []
    lines.append('\nAdjusted Rand Index (ARI): {:.4f}'.format(metrics['clustering']['ARI']))
    lines.append('Normalized Mutual Information (NMI): {:.4f}'.format(metrics['clustering']['NMI']))
    lines.append('Unsupervised Clustering Accuracy (ACC): {:.4f}'.format(metrics['clustering']['ACC']))
    logging.info('\n'.join(lines))

def print_fdd(metrics, logging):
    lines = []
    lines.append('\nTPR/FPR:')
    for i in np.arange(1, len(metrics['classification']['TPR'])).astype('int'):
        lines.append('    Fault {:02d}: {:.4f}/{:.4f}'.format(i, metrics['classification']['TPR'][i], metrics['classification']['FPR'][i]))

    lines.append('Detection TPR: {:.4f}'.format(metrics['detection']['TPR']))
    lines.append('Detection FPR: {:.4f}'.format(metrics['detection']['FPR']))
    lines.append('Average Detection Delay (ADD): {:.2f}'.format(metrics['detection']['ADD']))
    lines.append('Total Correct Diagnosis Rate (Total CDR): {:.4f}'.format(metrics['diagnosis']['CDR_total']))
    logging.info('\n'.join(lines))
