import gc

import numpy as np
import pandas as pd
from cuml.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score
import torch
from torch.utils.data import DataLoader
from fddbenchmark import FDDDataloader, FDDDataset
from tqdm import tqdm, trange
import scipy
import logging

from models.sensordbscan.utils import pairwise_euclidean_distance
from models.sensordbscan.visualization_utils import visualize_all
from utils import build_costs_matrix


def build_pretraining_dataloader(cfg):
    dataset = _build_fdd_dataset(cfg)

    train_dataset = PretraingDataset(cfg, dataset)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=cfg.train_batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=4, pin_memory=True)

    return train_dataloader


def add_smaller_windows(X, y, embs, pad_masks, cfg, model):
    expanded_X = X.clone().cpu()
    expanded_y = y.clone().cpu()
    expanded_embs = embs.clone().cpu()

    step_size = int(cfg.step_size / cfg.train_fraction)

    for window_size in sorted(cfg.window_sizes)[:-1]:
        ws_multiplier = (max(cfg.window_sizes) - window_size) // step_size + 1

        w_pad_masks = torch.ones((X.shape[0] * ws_multiplier, X.shape[1]), dtype=torch.bool, device=cfg.device)
        w_pad_masks[:, window_size:] = 0
        w_X = torch.zeros((X.shape[0] * ws_multiplier, X.shape[1], X.shape[2]), dtype=torch.float32, device=cfg.device)
        w_y = y.repeat(ws_multiplier).cpu()
        w_embs = torch.empty((X.shape[0] * ws_multiplier, embs.shape[1]), dtype=torch.float32, device='cpu')

        for i in range(ws_multiplier):
            s = i * step_size
            w_X[i * X.shape[0]: (i + 1) * X.shape[0], :window_size] = X[:, s: s + window_size].to(cfg.device)

        for i, (batch_X, batch_masks) in enumerate(zip(w_X.split(cfg.batch_size), w_pad_masks.split(cfg.batch_size))):
            with torch.no_grad():
                pred = model(batch_X, batch_masks)[1]

            w_embs[i * cfg.batch_size: (i + 1) * cfg.batch_size] = pred.cpu()

        expanded_X = torch.cat([expanded_X, w_X.cpu()], dim=0)
        expanded_y = torch.cat([expanded_y, w_y], dim=0)
        expanded_embs = torch.cat([expanded_embs, w_embs], dim=0)
        pad_masks = torch.cat([pad_masks, w_pad_masks.cpu()], dim=0)

        assert expanded_embs.shape[0] == expanded_y.shape[0] == expanded_X.shape[0] == pad_masks.shape[0]

    return expanded_X, expanded_y, expanded_embs, pad_masks


def build_triplets_loader(cfg, slices_dataset, model, indices, ch_scores, epoch, prev_loss=0):
    # dataloader = DataLoader(slices_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    gc.collect()
    torch.cuda.empty_cache()

    embs = list()
    ys = list()
    pad_mask = torch.ones((cfg.eval_batch_size, max(cfg.window_sizes)), dtype=torch.bool, device=cfg.device)
    for X, _, y in tqdm(slices_dataset, desc=f'Epoch #{epoch}. Calculating embeddings'):
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.int32)
        with torch.no_grad():
            pred = model(X.to(cfg.device), pad_mask[:X.shape[0]])

        embs.append(pred[1])
        ys.extend(y.numpy().tolist())

    embs = torch.cat(embs, dim=0)
    # with open('embeds.npy', 'wb') as f:
    #     np.save(f, embs.cpu().numpy())

    gc.collect()
    torch.cuda.empty_cache()

    min_samples = int(embs.shape[0] * cfg.min_cluster_fraction)
    clustering_labels = DBSCAN(eps=cfg.epsilon*cfg.dbscan_epsilon_multiplier, min_samples=min_samples,
                               metric=cfg.metric, max_mbytes_per_batch=cfg.max_mbytes_per_batch).fit_predict(embs.cpu().numpy())

    outliers_factor = np.sum(clustering_labels == -1) / embs.shape[0]

    nclusters = np.unique(clustering_labels).shape[0]

    score = calinski_harabasz_score(embs.cpu().numpy(), clustering_labels) if nclusters > 1 else 0

    logging.info(f'Epoch #{epoch}. Calinski-Harabasz score: {round(score, 2)}, #clusters: {nclusters}, outliers factor: {round(outliers_factor, 6)}')

    selected_indices = None
    # if ((len(ch_scores) < 1 or score * cfg.ch_score_momentum <= ch_scores[-1] or outliers_factor > 0.01)
    #         and epoch < cfg.epochs):
    if True:
        selected_indices = select_samples_to_label(embs.cpu().numpy(), clustering_labels,
                                                   np.array([ys[i] for i in indices]),
                                                   cfg.n_samples_to_select, indices, epoch)

    if cfg.visualize:
        visualize_all(embs.cpu().numpy(), clustering_labels, ys, selected_indices, cfg, epoch)

    # TODO: add number of samples logging
    if selected_indices is not None:
        indices = np.concatenate([indices, selected_indices])
        indices = torch.IntTensor(indices)

    ch_scores.append(score)

    # xy_pairs = [slices_dataset[i // cfg.eval_batch_size] for i in indices]
    # X_ = torch.stack([torch.as_tensor(xy_pairs[c][0][i % cfg.eval_batch_size], dtype=torch.float32)
    #                   for c, i in enumerate(indices)], dim=0)
    # y_ = torch.IntTensor([xy_pairs[c][2][i % cfg.eval_batch_size] for c, i in enumerate(indices)])
    X_ = torch.stack([torch.as_tensor(slices_dataset[i // cfg.eval_batch_size][0][i % cfg.eval_batch_size],
                                      dtype=torch.float32) for i in indices], dim=0)
    y_ = torch.IntTensor([slices_dataset[i // cfg.eval_batch_size][2][i % cfg.eval_batch_size] for i in indices])
    embs_ = embs[indices]
    pad_masks = torch.ones(*X_.shape[:-1], dtype=torch.bool, device='cpu')

    del embs
    gc.collect()
    torch.cuda.empty_cache()

    # TODO: make smaller windows with variable padding position
    if len(cfg.window_sizes) > 1:
        X_, y_, embs_, pad_masks = add_smaller_windows(X_.cpu(), y_.cpu(), embs_.cpu(), pad_masks, cfg, model)

    triplets_dataset = TripletsDataset(X_, y_, embs_, pad_masks, cfg)
    triplets_loader = torch.utils.data.DataLoader(dataset=triplets_dataset,
                                                  batch_size=cfg.batch_size,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True)

    if triplets_dataset.triplet_idxs.shape[0] < 10:
        return build_triplets_loader(cfg, slices_dataset, model, indices, ch_scores, epoch)

    return triplets_loader, indices, ch_scores


class PretraingDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, fdd_dataset):
        self.fdd_dataset = fdd_dataset

        self.fddloader = FDDDataloader(
            dataframe=self.fdd_dataset.df,
            label=self.fdd_dataset.label,
            mask=self.fdd_dataset.train_mask,
            window_size=cfg.dataset_window_size,
            step_size=cfg.step_size,
            use_minibatches=True,
            batch_size=1,
            shuffle=False
        )

        self.augmentor = Augmentor(cfg.augmentor)

    def __len__(self):
        return len(self.fddloader)

    def __getitem__(self, idx):
        window, _, label = self.fddloader[idx]
        X_weak, targets_weak, target_masks_weak, X_strong, targets_strong, target_masks_strong = self.augmentor(
            window[0])

        label = torch.LongTensor([label[0].astype('int')])

        return X_weak, targets_weak, target_masks_weak, X_strong, targets_strong, target_masks_strong, label


class SlicesDataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, mask: pd.Series, window_size: int, step_size: int):
        super(SlicesDataset, self).__init__()

        self.window_size = window_size

        self.X = X
        self.y = y
        self.windows_end_indices = list()

        run_ids = self.X[mask].index.get_level_values(0).unique()
        for run_id in tqdm(run_ids, desc='Creating sequence of samples'):
            indices = np.array(self.X.index.get_locs([run_id]))
            indices = indices[self.window_size - 1:]
            indices = indices[::step_size]
            indices = indices[mask.iloc[indices].to_numpy(dtype=bool)]
            self.windows_end_indices.extend(indices)

        self.windows_end_indices = torch.IntTensor(self.windows_end_indices)
        self.X = torch.FloatTensor(self.X.to_numpy())
        self.y = torch.IntTensor(self.y.to_numpy())

    def __len__(self):
        return len(self.windows_end_indices)

    def __getitem__(self, idx):
        window_end_idx = self.windows_end_indices[idx]
        X_window = self.X[window_end_idx - self.window_size + 1:window_end_idx + 1]
        y_window = self.y[window_end_idx - self.window_size + 1:window_end_idx + 1]
        y = torch.mode(y_window).values.max(dim=0).values.item()
        return torch.FloatTensor(X_window), y


class TripletsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, embs, pad_masks, cfg):
        super().__init__()

        self.X = X
        self.y = y
        self.pad_masks = pad_masks

        self.triplet_idxs = torch.tensor([], dtype=torch.int32, device=cfg.device)
        self.triplet_distances = torch.tensor([], dtype=torch.float32, device=cfg.device)
        self.cfg = cfg

        if cfg.metric == 'euclidean':
            # distance_matrix = torch.nn.functional.pairwise_distance(embs.unsqueeze(0), embs.unsqueeze(1)).to(cfg.device)
            distance_matrix = pairwise_euclidean_distance(embs.to(cfg.device), embs.to(cfg.device))
        elif cfg.metric == 'cosine':
            distance_matrix = torch.nn.functional.cosine_similarity(embs.unsqueeze(0),
                                                                    embs.unsqueeze(1), dim=-1).to(cfg.device)
        else:
            raise ValueError(f'got unexpected distance metric: {cfg.metric}')

        distance_matrix = distance_matrix.triu().to(cfg.device)

        # TODO: add logging in the following part
        # 2. construct, filter and sort triplet indices
        for l in tqdm(torch.unique(self.y), desc='Constructing triplets'):
            positive_idxs = (self.y == l).nonzero()[:, 0]
            negative_idxs = (self.y != l).nonzero()[:, 0]

            positive_idxs = positive_idxs.to(cfg.device)
            negative_idxs = negative_idxs.to(cfg.device)

            for i, aidx in enumerate(positive_idxs):
                # (P,) -> (P, N)
                ap_distances = distance_matrix[aidx, positive_idxs[i+1:]].unsqueeze(1).tile((1, len(negative_idxs)))
                # (N,) -> (1, N)
                an_distances = distance_matrix[aidx, negative_idxs].unsqueeze(0)
                # (P, N)
                triplet_distances_ = an_distances - ap_distances
                triplet_distances_[triplet_distances_ < 0] = 0

                pn_indices = triplet_distances_.nonzero().to(cfg.device)
                pn_indices = torch.stack([positive_idxs[i+1:][pn_indices[:, 0]],
                                          negative_idxs[pn_indices[:, 1]]], dim=1)

                triplet_idxs = torch.cat([torch.full((triplet_distances_.nonzero().shape[0], 1), aidx, device=cfg.device),
                                          pn_indices], dim=1)

                self.triplet_idxs = torch.cat([self.triplet_idxs, triplet_idxs])
                self.triplet_distances = torch.cat([self.triplet_distances, triplet_distances_[triplet_distances_ > 0]])

                # 3. take N best (N smallest abs values)
                max_triplets = cfg.n_triplet_batches * cfg.batch_size
                self.triplet_idxs = self.triplet_idxs[torch.argsort(self.triplet_distances).to(cfg.device)][:max_triplets]
                self.triplet_distances = self.triplet_distances[torch.argsort(self.triplet_distances)][:max_triplets]

        self.triplet_idxs = self.triplet_idxs.cpu()

    def __getitem__(self, idx):
        aidx, pidx, nidx = self.triplet_idxs[idx]
        return (self.X[aidx], self.X[pidx], self.X[nidx],
                self.pad_masks[aidx], self.pad_masks[pidx], self.pad_masks[nidx])

    def __len__(self):
        return len(self.triplet_idxs)


def select_samples_to_label(embs, clustering_labels, known_ys, number_samples_to_select, old_indices, epoch):
    unique_clusters, cluster_sizes = np.unique(clustering_labels, return_counts=True)

    if len(old_indices) > 0:
        pred_ys = clustering_labels[old_indices]
        if pred_ys.min() < 0:
            pred_ys += 1

        # For each cluster find number of labels minus most frequent one
        # (so we may estimate intra-cluster dispersion of targets for each of clusters and assign sampling weights
        # accordingly)
        costs_matrix = build_costs_matrix(known_ys, pred_ys, nclusters=unique_clusters.shape[0])
        costs_matrix[costs_matrix.argmax(axis=0), np.arange(costs_matrix.shape[1])] = 0

        if costs_matrix.sum() == 0:
            weights = cluster_sizes / cluster_sizes.sum()
        else:
            weights = np.log(costs_matrix.sum(axis=0) + costs_matrix.mean())
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
    else:
        weights = cluster_sizes / cluster_sizes.sum()

    samples_indices = list()
    pbar = trange(number_samples_to_select, desc=f'Epoch #{epoch}. Selecting {number_samples_to_select} samples to label')
    while len(samples_indices) < number_samples_to_select:
        cluster = np.random.choice(unique_clusters, p=weights)

        cluster_indices = np.where(clustering_labels == cluster)[0]
        cluster_indices = np.setdiff1d(cluster_indices, old_indices)
        cluster_indices = np.setdiff1d(cluster_indices, samples_indices)
        cluster_samples = embs[cluster_indices]

        # otherwise empty clusters will cause an error
        if len(cluster_samples) == 0:
            weights[unique_clusters == cluster] = 0
            if weights.sum() == 0:
                logging.info(f'Epoch #{epoch}. All clusters are empty')
                break
            weights = weights / weights.sum()
            continue

        cluster_mean = np.mean(cluster_samples)
        cluster_std = np.std(cluster_samples)

        probs = scipy.stats.norm.pdf(np.mean(cluster_samples, axis=-1), cluster_mean, cluster_std)
        probs /= sum(probs)

        selected_index = np.random.choice(cluster_indices, p=probs)
        samples_indices.append(selected_index)

        pbar.update(1)

    return np.array(samples_indices).astype(int)


def _build_fdd_dataset(cfg, big_cluster_reduction=False):
    dataset = FDDDataset(name=cfg.dataset_name)
    dataset.df.drop(['xmeas_23', 'xmeas_24', 'xmeas_25', 'xmeas_26',
                     'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_30', 'xmeas_31', 'xmeas_32',
                     'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36', 'xmeas_37', 'xmeas_38',
                     'xmeas_39', 'xmeas_40', 'xmeas_41', 'xmv_5', 'xmv_9', 'xmeas_19'], axis=1, inplace=True)

    mean = dataset.df[dataset.train_mask].mean(axis=0)
    std = dataset.df[dataset.train_mask].std(axis=0)
    dataset.df = (dataset.df - mean) / std

    if cfg.dataset_name == "rieth_tep":
        dataset.train_mask = make_rieth_imbalance(dataset.train_mask, big_cluster_reduction)

    return dataset


class Augmentor():

    def __init__(self, cfg):
        super(Augmentor, self).__init__()

        self.masking_ratio = cfg.masking_ratio
        self.mean_mask_length = cfg.mean_mask_length

        self.weak_jitter_sigma = cfg.weak_jitter_sigma
        self.weak_scaling_loc = cfg.weak_scaling_loc
        self.weak_scaling_sigma = cfg.weak_scaling_sigma
        self.strong_scaling_loc = cfg.strong_scaling_loc
        self.strong_scaling_sigma = cfg.strong_scaling_sigma
        self.strong_permute_max_segments = cfg.strong_permute_max_segments

    def __call__(self, X):
        X = X.transpose(1, 0)  # (feat_dim, seq_length) array
        weak = jitter(scaling(X, loc=self.weak_scaling_loc, sigma=self.weak_scaling_sigma),
                      sigma=self.weak_jitter_sigma)
        strong = scaling(permutation(X, max_segments=self.strong_permute_max_segments), loc=self.strong_scaling_loc,
                         sigma=self.strong_scaling_sigma)

        weak = weak.transpose(1, 0)  # (seq_length, feat_dim) array
        strong = strong.transpose(1, 0)  # (seq_length, feat_dim) array
        X = X.transpose(1, 0)

        mask_weak = noise_mask(weak, self.masking_ratio, self.mean_mask_length)  # (seq_length, feat_dim) boolean array
        mask_strong = noise_mask(strong, self.masking_ratio,
                                 self.mean_mask_length)  # (seq_length, feat_dim) boolean array

        X_weak, target_masks_weak = torch.FloatTensor(weak), torch.BoolTensor(mask_weak)
        X_strong, target_masks_strong = torch.FloatTensor(strong), torch.BoolTensor(mask_strong)

        targets_weak = X_weak.clone()
        X_weak = X_weak * target_masks_weak  # mask input
        X_weak = compensate_masking(X_weak, target_masks_weak)

        targets_strong = X_strong.clone()
        X_strong = X_strong * target_masks_strong
        X_strong = compensate_masking(X_strong, target_masks_strong)

        target_masks_weak = ~target_masks_weak  # inverse logic: 0 now means ignore, 1 means predict
        target_masks_strong = ~target_masks_strong

        return X_weak, targets_weak, target_masks_weak, X_strong, targets_strong, target_masks_strong


def noise_mask(X, masking_ratio, lm=3):
    mask = np.tile(np.expand_dims(geom_noise_mask_single(X.shape[0], lm, masking_ratio), 1), X.shape[1])

    return mask


def geom_noise_mask_single(L, lm, masking_ratio):
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (
                1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]

    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state

    return keep_mask


def compensate_masking(X, mask):
    # number of unmasked elements of feature vector for each time step
    num_active = torch.sum(mask, dim=-1).unsqueeze(-1)  # (batch_size, seq_length, 1)
    # to avoid division by 0, set the minimum to 1
    num_active = torch.max(num_active, torch.ones(num_active.shape, dtype=torch.int16))  # (batch_size, seq_length, 1)
    return X.shape[-1] * X / num_active


def jitter(x, sigma):  # 0.08
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, loc, sigma):  # 0.1
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=loc, scale=sigma, size=(1, x.shape[1]))

    return x * factor


def permutation(x, max_segments):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments)

    if num_segs > 1:
        split_points = np.random.choice(x.shape[1] - 2, num_segs - 1, replace=False)
        split_points.sort()
        splits = np.split(orig_steps, split_points)

        splits_indices = np.arange(len(splits))
        splits_indices = np.random.permutation(splits_indices)

        # TODO: open an issue
        # np.random.permute over splits gave size error
        warp = np.concatenate([splits[idx] for idx in splits_indices]).ravel()
        return x[:, warp]
    else:
        return x


def make_rieth_imbalance(train_mask, simulate_cluster_reduction=False):
    # make rieth_tep imbalance: 500 normal runs and 5 runs for each fault
    train_runs = train_mask[train_mask].index.get_level_values(0).unique()

    if simulate_cluster_reduction:
        imbalance_train_runs = list(train_runs[:25]) + list(train_runs[500:600])
    else:
        imbalance_train_runs = train_runs[:600]

    imbalance_train_mask = train_mask.copy()
    imbalance_train_mask.loc[:] = False
    imbalance_train_mask.loc[imbalance_train_runs] = True
    return imbalance_train_mask
