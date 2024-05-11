import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.manifold import TSNE
from cuml import TSNE, UMAP
import numpy as np
import matplotlib

matplotlib.use('Agg')


def show_fig_with_timer(fig, interval):
    def close_event():
        plt.close()

    timer = fig.canvas.new_timer(interval)
    timer.add_callback(close_event)

    timer.start()
    plt.show()


def visualize_all(embs, clustering_labels, gt_labels, selected_indices, cfg, epoch):
    umap_decomposed = UMAP(n_components=2, metric=cfg.metric).fit_transform(embs)
    tsne_decomposed = TSNE(n_components=2, metric=cfg.metric).fit_transform(embs)

    # TODO: match labels so scatterplot colors are the same
    # label_matching = label_assignment(gt_labels, clustering_labels)
    # gt_labels_matched = gt_labels.copy()
    # for old_value, new_value in enumerate(label_matching):
    #     gt_labels_matched[gt_labels == old_value] = new_value

    visualize_clustering(umap_decomposed, tsne_decomposed, clustering_labels, gt_labels, epoch)
    if selected_indices is not None:
        visualize_selected_for_labelling(umap_decomposed, tsne_decomposed, selected_indices, epoch)


def visualize_clustering(umap_decomposed, tsne_decomposed, clustering_labels, gt_labels, epoch):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 12))

    axes[0, 0].scatter(umap_decomposed[:, 0], umap_decomposed[:, 1], s=1, c=clustering_labels)
    axes[0, 1].scatter(tsne_decomposed[:, 0], tsne_decomposed[:, 1], s=1, c=clustering_labels)

    axes[0, 0].set_title('UMAP decomposition')
    axes[0, 1].set_title('TSNE decomposition')

    axes[1, 0].scatter(umap_decomposed[:, 0], umap_decomposed[:, 1], s=1, c=gt_labels)
    axes[1, 1].scatter(tsne_decomposed[:, 0], tsne_decomposed[:, 1], s=1, c=gt_labels)

    axes[1, 0].set_title('UMAP decomposition')
    axes[1, 1].set_title('TSNE decomposition')

    fig.suptitle(f'Epoch #{epoch}, got {np.unique(clustering_labels).shape[0]} clusters')
    fig.savefig(f'./visualization/clustering_{epoch}.png')
    plt.close(fig)


def visualize_selected_for_labelling(umap_decomposed, tsne_decomposed, indices, epoch):
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))

    axes[0].scatter(umap_decomposed[:, 0], umap_decomposed[:, 1], s=1, c='lightgreen')
    axes[1].scatter(tsne_decomposed[:, 0], tsne_decomposed[:, 1], s=1, c='lightgreen')

    axes[0].scatter(umap_decomposed[indices, 0], umap_decomposed[indices, 1], s=10, c='red')
    axes[1].scatter(tsne_decomposed[indices, 0], tsne_decomposed[indices, 1], s=10, c='red')

    axes[0].set_title('UMAP decomposition')
    axes[1].set_title('TSNE decomposition')

    fig.savefig(f'./visualization/sampling_{epoch}.png')
    plt.close(fig)
