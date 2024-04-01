import hydra
from models import pca_kmeans, st_catgan, convae, sensorscan, sensordbscan
from utils import weighted_max_occurence, print_clustering, print_fdd
import logging
from fddbenchmark import FDDEvaluator
import pandas as pd
import numpy as np


@hydra.main(version_base=None, config_path="configs")
def main(cfg):

    if cfg.model == 'pca_kmeans':
        train_pred, train_label, test_pred, test_label = pca_kmeans.run(cfg)
    elif cfg.model == 'st_catgan':
        train_pred, train_label, test_pred, test_label = st_catgan.run(cfg) 
    elif cfg.model == 'convae':
        train_pred, train_label, test_pred, test_label = convae.run(cfg) 
    elif cfg.model == 'sensorscan':
        train_pred, train_label, test_pred, test_label = sensorscan.run(cfg)
    elif cfg.model == 'sensordbscan':
        train_pred, train_label, test_pred, test_label = sensordbscan.run(cfg)
    else:
        raise NotImplementedError

    if cfg.classes == 'all':
        if cfg.dataset == 'small_tep':
            n_types = 21
        elif cfg.dataset == 'rieth_tep':
            n_types = 21
        elif cfg.dataset == 'reinartz_tep':
            n_types = 29
        else:
            raise NotImplementedError(f'Got unknown dataset: {cfg.dataset}')
    else:
        n_types = len(cfg.classes)

    logging.info(f'Got {np.unique(test_pred).shape[0]} clusters')

    logging.info('Calculating clustering metrics')
    evaluator = FDDEvaluator(step_size=cfg.step_size)
    metrics = evaluator.evaluate(test_label, test_pred)
    print_clustering(metrics, logging)

    logging.info('Creating label matching')
    label_matching = weighted_max_occurence(test_label, test_pred, n_types)

    test_pred = pd.Series(label_matching[test_pred], index=test_pred.index)
    logging.info('Calculating FDD metrics')
    evaluator = FDDEvaluator(step_size=cfg.step_size)
    metrics = evaluator.evaluate(test_label, test_pred)
    print_fdd(metrics, logging)


if __name__ == '__main__':
    main()
