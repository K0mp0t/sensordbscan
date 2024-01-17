import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import torch
from fddbenchmark import FDDDataloader, FDDDataset
from tqdm import tqdm
import scipy
import logging


def build_pretraining_dataloader(cfg):
    dataset = _build_fdd_dataset(cfg)

    train_dataset = PretraingDataset(cfg, dataset)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=cfg.train_batch_size,
                                                   shuffle=True,
                                                   drop_last = True,
                                                   num_workers=6, pin_memory=True)

    return train_dataloader


def build_neighbour_loader(cfg, encoder):
    dataset = _build_fdd_dataset(cfg.pretraining, True)

    train_dataset = TrainingDataset(cfg.pretraining, dataset)
    train_subsample_idxs = np.random.choice(np.arange(0, len(train_dataset)),
                                            min(len(train_dataset), cfg.neighbour_dataset_size), replace=False)
    train_dataset_subset = torch.utils.data.Subset(train_dataset, train_subsample_idxs)

    neighbor_dataset = NeighborDataset.build_with_encoder(cfg, train_dataset_subset, encoder)

    neighbor_loader = torch.utils.data.DataLoader(dataset=neighbor_dataset,
                                                  batch_size=cfg.scan_batch_size,
                                                  shuffle=True,
                                                  num_workers=4,
                                                  pin_memory=True, drop_last=True)

    return neighbor_loader


def build_triplets_loader(cfg, sliced_dataset, model, old_indices):
    # dataset = _build_fdd_dataset(cfg.pretraining, False)
    dataloader = torch.utils.data.DataLoader(sliced_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                             num_workers=6, pin_memory=True)

    embs = list()
    for X, _ in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            pad_mask = torch.ones(*X.shape[:-1], dtype=torch.bool, device=cfg.device)
            pred = model(X.to(cfg.device), pad_mask)
            embs.append(pred[1])

    embs = torch.cat(embs, dim=0)
    clustering_labels = DBSCAN().fit_predict(embs.cpu().numpy())

    indices = select_samples_to_label(embs.cpu(), clustering_labels, cfg.n_samples_to_select, old_indices)

    triplets_dataset = TripletsDataset(sliced_dataset.X[indices], sliced_dataset.y[indices], embs[indices], cfg.max_triplets)
    triplets_loader = torch.utils.data.DataLoader(dataset=triplets_dataset,
                                                  batch_size=cfg.batch_size,
                                                  shuffle=True,
                                                  num_workers=6, pin_memory=True)

    return triplets_loader, indices


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
    def __init__(self, X: np.ndarray, y: np.ndarray, window_size: int):
        super(SlicesDataset, self).__init__()

        self.window_size = window_size
        self.X = X  # (N, features)
        self.y = y  # (N,)

        nmax = self.X.shape[0] // window_size * window_size
        self.X = self.X[:nmax]
        self.y = self.y[:nmax]

        self.X = self.X.reshape(-1, self.window_size, self.X.shape[-1])
        self.y = self.y.reshape(-1, self.window_size)

        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TripletsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, embs, max_triplets, metric='euclidean'):
        super().__init__()

        self.X = X
        self.y = y

        self.y = torch.mode(self.y, dim=-1).values

        self.idxs_triplets = torch.IntTensor()
        self.triplet_distances = torch.FloatTensor()

        distance_matrix = torch.zeros((X.shape[0], X.shape[0]))

        # 1. calculate pairwise distance
        for i in range(X.shape[0]):
            for j in range(i, X.shape[1]):  # save a little computation time
                if metric == 'euclidean':
                    distance_matrix[i, j] = torch.mean(torch.abs(embs[i] - embs[j]))
                elif metric == 'cosine':
                    distance_matrix[i, j] = torch.nn.functional.cosine_similarity(embs[i], embs[j], dim=0)
                else:
                    raise ValueError(f'got unexpected distance metric: {metric}')

        # 2. construct, filter and sort triplet indices
        for l in torch.unique(self.y):
            positive_idxs = (self.y == l).nonzero()
            negative_idxs = (self.y != l).nonzero()

            for aidx in positive_idxs:
                for pidx in positive_idxs[aidx+1:]:
                    ap_distance = distance_matrix[aidx, pidx]

                    an_distances = distance_matrix[aidx, negative_idxs]
                    an_pairs = torch.IntTensor([(aidx, nidx) for nidx in negative_idxs])

                    triplet_distances_ = an_distances - ap_distance

                    # we select semi-hard triplets to make training process more stable
                    an_pairs = an_pairs[triplet_distances_[:, 0] > 0, :]
                    triplet_distances_ = triplet_distances_[triplet_distances_ > 0]

                    self.idxs_triplets = torch.cat([self.idxs_triplets, torch.IntTensor([[aidx, pidx, ni] for ni in an_pairs[:, 1]])])
                    self.triplet_distances = torch.cat([self.triplet_distances, triplet_distances_])

        # 3. take N best (N smallest abs values)
        self.idxs_triplets = self.idxs_triplets[torch.argsort(torch.abs(self.triplet_distances))][:max_triplets]

    def __getitem__(self, idx):
        aidx, pidx, nidx = self.idxs_triplets[idx]
        return self.X[aidx], self.X[pidx], self.X[nidx]

    def __len__(self):
        return len(self.idxs_triplets)


def select_samples_to_label(embs, clustering_labels, number_samples_to_select, old_indices):
    unique_clusters, cluster_sizes = np.unique(clustering_labels, return_counts=True)
    cluster_sizes = cluster_sizes[unique_clusters >= 0]
    unique_clusters = unique_clusters[unique_clusters >= 0]

    samples_indices = list()
    while len(samples_indices) < number_samples_to_select:
        cluster = np.random.choice(unique_clusters, p=cluster_sizes/sum(cluster_sizes))

        cluster_indices = np.where(clustering_labels == cluster)[0]
        cluster_indices = np.setdiff1d(cluster_indices, old_indices)
        cluster_indices = np.setdiff1d(cluster_indices, samples_indices)
        cluster_samples = embs[cluster_indices]

        cluster_mean = torch.mean(cluster_samples)
        cluster_std = torch.std(cluster_samples)

        probs = [scipy.stats.norm.pdf(torch.mean(s), cluster_mean, cluster_std) for s in cluster_samples]
        probs /= sum(probs)

        selected_index = np.random.choice(cluster_indices, p=probs)
        samples_indices.append(selected_index)

    return np.concatenate([samples_indices, old_indices]).astype(int)


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, fdd_dataset):
        self.fdd_dataset = fdd_dataset
        self.train = True
        self.fddloader = FDDDataloader(
            dataframe=self.fdd_dataset.df,
            label=self.fdd_dataset.label,
            mask=self.fdd_dataset.train_mask,
            window_size=cfg.model_input_length,
            step_size=cfg.step_size,
            use_minibatches=True,
            batch_size=1,
            shuffle=False
        )

    def __len__(self):
        return len(self.fddloader)

    def __getitem__(self, idx):
        window, _, label = self.fddloader[idx]

        window = torch.FloatTensor(window[0])
        label = torch.LongTensor([label[0].astype('int')])

        return window, label


class NeighborDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, neighbor_indices):
        super(NeighborDataset, self).__init__()

        self.dataset = dataset
        self.neighbor_indices = neighbor_indices

    @classmethod
    def build_with_encoder(cls, cfg, dataset, encoder):
        encoder = encoder.eval()

        embedings_list = []

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=cfg.scan_batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True, drop_last=False)

        for X, _ in tqdm(loader, total=len(loader)):
            X = X.to(cfg.device)
            pad_mask = torch.ones(*X.shape[:-1], dtype=torch.bool, device=cfg.device)

            with torch.no_grad():
                _, embeddings = encoder(X, pad_mask)

            embedings_list.append(embeddings.cpu())

        embeddings = torch.cat(embedings_list, dim=0).numpy()

        nbrs = NearestNeighbors(n_neighbors=cfg.num_neighbors, algorithm='ball_tree').fit(embeddings)
        _, idxs = nbrs.kneighbors(embeddings)
        idxs = idxs[:, 1:]

        return cls(dataset, idxs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, label = self.dataset[idx]

        neighbor_ind = np.random.choice(self.neighbor_indices[idx])
        neighbor, _ = self.dataset[neighbor_ind]

        return X, neighbor, label


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
