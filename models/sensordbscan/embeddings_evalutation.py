import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from sklearn.metrics import calinski_harabasz_score


def evaluate_embeddings(embs, epsilon, metric='euclidean'):
    clustering_labels = DBSCAN(eps=epsilon, min_samples=1, metric=metric).fit_predict(embs.detach().cpu().numpy())
    outliers_factor = 1 - np.sum(clustering_labels == -1) / embs.shape[0]

    cmask = clustering_labels > -1

    return calinski_harabasz_score(embs[cmask], clustering_labels[cmask]) * outliers_factor
