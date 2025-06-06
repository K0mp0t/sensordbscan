import torch
import numpy as np
import os

from torch.utils.data import DataLoader

from models.sensordbscan.data_utils import build_pretraining_dataloader, build_triplets_loader, SlicesDataset
from models.sensordbscan.model import build_encoder, build_clustering, SensorSCAN, SensorDBSCAN
from models.sensordbscan.optim import build_pretraining_optim, build_scan_optim, build_triplet_optim
from models.sensordbscan.train_utils import train_ssl_epoch, train_scan_epoch, train_triplet_epoch
from tqdm.auto import tqdm
import pandas as pd
from fddbenchmark import FDDDataset, FDDDataloader
import logging
from utils import exclude_columns, make_rieth_imbalance, normalize, exclude_classes, take_dataset_fraction
import gc
import cuml

cuml.set_global_output_type('numpy')


def run(cfg):
    
    dataset_name = cfg.dataset
    window_size = max(cfg.window_sizes)
    step_size = cfg.step_size
    random_seed = cfg.random_seed
    eval_batch_size = cfg.eval_batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.device = device
    cfg.pretraining.device = device

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    logging.info('Creating dataset')

    dataset = FDDDataset(name=dataset_name)
    dataset.df = exclude_columns(dataset.df)

    exclude_classes(dataset, cfg.classes)

    normalize(dataset)
    # if dataset_name == 'rieth_tep':
    #     dataset.train_mask = make_rieth_imbalance(dataset.train_mask)

    dataset = take_dataset_fraction(dataset, cfg)

    logging.info('Creating dataloaders')

    train_loader = FDDDataloader(
        dataframe=dataset.df, 
        mask=dataset.train_mask, 
        label=dataset.label,
        window_size=window_size, 
        step_size=step_size,
        use_minibatches=True, 
        batch_size=eval_batch_size,
        shuffle=True,
    )

    test_loader = FDDDataloader(
        dataframe=dataset.df, 
        mask=dataset.test_mask, 
        label=dataset.label,
        window_size=window_size, 
        step_size=step_size,
        use_minibatches=True, 
        batch_size=eval_batch_size,
        shuffle=False,
    )

    # TODO: improve logging: add CH score, outliers factor and number of clusters graphs

    if cfg.path_to_model is None:
        encoder = build_encoder(cfg.pretraining)
        if cfg.path_to_encoder is not None and os.path.exists(cfg.path_to_encoder):
            logging.info('Using pretrained encoder')
            encoder.load_state_dict(torch.load(cfg.path_to_encoder))
        else:
            logging.info('Pretraining encoder')
            pretraining_loader = build_pretraining_dataloader(cfg.pretraining)
            loss_fn, optimizer = build_pretraining_optim(cfg.pretraining, encoder)
            for epoch in range(cfg.pretraining.epochs):
                avg_loss = train_ssl_epoch(cfg.pretraining, encoder, pretraining_loader, loss_fn, optimizer)
                logging.info(f'Epoch {epoch}: loss = {avg_loss:10.8f}')

            torch.save(encoder.state_dict(), f'./saved_models/pretrained_encoder_{cfg.dataset}_rope_sp.pth')

        logging.info('Training encoder with triplet loss')
        indices = list()
        ch_scores = list()

        slices_dataset = train_loader
        loss_fn, optimizer = build_triplet_optim(cfg, encoder)

        avg_loss = 0
        for epoch in range(cfg.epochs):
            triplets_loader, indices, ch_scores = build_triplets_loader(cfg, slices_dataset, encoder, indices,
                                                                        ch_scores, epoch, avg_loss)
            avg_loss = train_triplet_epoch(cfg, encoder, triplets_loader, loss_fn, optimizer)
            logging.info(f'Epoch #{epoch}. loss = {avg_loss:10.8f}')

        cfg.path_to_model = f'saved_models/sensordbscan_encoder_{dataset_name}.pth'
        torch.save(encoder.state_dict(), cfg.path_to_model)

    gc.collect()
    torch.cuda.empty_cache()

    encoder = build_encoder(cfg.pretraining)
    encoder.load_state_dict(torch.load(cfg.path_to_model, map_location=cfg.device))

    sensordbscan = SensorDBSCAN(encoder, cfg, avg_loss)

    logging.info('Getting predictions on train')
    encoder.eval()
    train_embs = []
    train_label = []

    for X, time_index, label in tqdm(train_loader, desc='Getting predictions on train'):
        X = torch.FloatTensor(X).to(cfg.device)
        with torch.no_grad():
            pred = sensordbscan.get_embs(X)
        train_embs.append(pred)
        train_label.append(pd.Series(label, index=time_index))
    #
    train_embs = torch.cat(train_embs, dim=0)
    train_label = pd.concat(train_label).astype('int')

    np.save(f'embeddings_{dataset_name}_train.npy', train_embs.cpu().numpy())
    # np.save(f'labels_{dataset_name}_train.npy', train_label)
    train_label.to_csv(f'labels_{dataset_name}_train.csv')
    np.save(f'train_labelled_indices_{dataset_name}.npy', indices.numpy())

    gc.collect()
    torch.cuda.empty_cache()

    train_pred = sensordbscan.cluster_embs(train_embs.cpu().numpy())
    train_pred = pd.Series(train_pred, index=train_label.index)

    logging.info('Getting predictions on test')
    test_embs = []
    test_label = []
    for X, time_index, label in tqdm(test_loader, desc='Getting predictions on test'):
        X = torch.FloatTensor(X).to(cfg.device)
        with torch.no_grad():
            pred = sensordbscan.get_embs(X)
        test_embs.append(pred)
        test_label.append(pd.Series(label, index=time_index))

    test_embs = torch.cat(test_embs, dim=0)
    test_label = pd.concat(test_label).astype('int')

    np.save(f'embeddings_{dataset_name}_test.npy', test_embs.cpu().numpy())
    # np.save(f'labels_{dataset_name}_test.npy', test_label)
    test_label.to_csv(f'labels_{dataset_name}_test.csv')

    gc.collect()
    torch.cuda.empty_cache()

    test_pred = sensordbscan.cluster_embs(test_embs.cpu().numpy())
    test_pred = pd.Series(test_pred, index=test_label.index)

    if cfg.visualize_final_state:
        from models.sensordbscan.visualization_utils import visualize_all
        visualize_all(train_embs.cpu().numpy(), train_pred, train_label, None, cfg, 'final_train')
        visualize_all(test_embs.cpu().numpy(), test_pred, test_label, None, cfg, 'final_test')

    return train_pred, train_label, test_pred, test_label
