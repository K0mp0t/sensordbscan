import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn.manifold import TSNE
from cuml import TSNE


def show_fig_with_timer(fig, interval):
    def close_event():
        plt.close()

    timer = fig.canvas.new_timer(interval)
    timer.add_callback(close_event)

    timer.start()
    plt.show()


def visualization_wrapper(embs, clustering_labels, gt_labels, selected_indices, cfg, epoch):
    pca_decomposed = PCA(n_components=2).fit_transform(embs)
    svd_decomposed = TruncatedSVD(n_components=2).fit_transform(embs)
    tsne_decomposed = TSNE(n_components=2, metric=cfg.metric, method='fft').fit_transform(embs)

    visualize_clustering(pca_decomposed, svd_decomposed, tsne_decomposed, clustering_labels, gt_labels, epoch)
    if selected_indices is not None:
        visualize_selected_for_labelling(pca_decomposed, svd_decomposed, tsne_decomposed, selected_indices, epoch)


def visualize_clustering(pca_decomposed, svd_decomposed, tsne_decomposed, clustering_labels, gt_labels, epoch):
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(15, 12))

    axes[0, 0].scatter(pca_decomposed[:, 0], pca_decomposed[:, 1], s=1, c=clustering_labels)
    axes[0, 1].scatter(svd_decomposed[:, 0], svd_decomposed[:, 1], s=1, c=clustering_labels)
    axes[0, 2].scatter(tsne_decomposed[:, 0], tsne_decomposed[:, 1], s=1, c=clustering_labels)

    axes[0, 0].set_title('PCA decomposition')
    axes[0, 1].set_title('SVD decomposition')
    axes[0, 2].set_title('TSNE decomposition')

    axes[1, 0].scatter(pca_decomposed[:, 0], pca_decomposed[:, 1], s=1, c=gt_labels)
    axes[1, 1].scatter(svd_decomposed[:, 0], svd_decomposed[:, 1], s=1, c=gt_labels)
    axes[1, 2].scatter(tsne_decomposed[:, 0], tsne_decomposed[:, 1], s=1, c=gt_labels)

    axes[1, 0].set_title('PCA decomposition')
    axes[1, 1].set_title('SVD decomposition')
    axes[1, 2].set_title('TSNE decomposition')

    fig.suptitle(f'Epoch #{epoch}')
    fig.savefig(f'./visualization/clustering_{epoch}.png')
    plt.close(fig)


def visualize_selected_for_labelling(pca_decomposed, svd_decomposed, tsne_decomposed, indices, epoch):
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 6))

    axes[0].scatter(pca_decomposed[:, 0], pca_decomposed[:, 1], s=1, c='lightgreen')
    axes[1].scatter(svd_decomposed[:, 0], svd_decomposed[:, 1], s=1, c='lightgreen')
    axes[2].scatter(tsne_decomposed[:, 0], tsne_decomposed[:, 1], s=1, c='lightgreen')

    axes[0].scatter(pca_decomposed[indices, 0], pca_decomposed[indices, 1], s=10, c='red')
    axes[1].scatter(svd_decomposed[indices, 0], svd_decomposed[indices, 1], s=10, c='red')
    axes[2].scatter(tsne_decomposed[indices, 0], tsne_decomposed[indices, 1], s=10, c='red')

    axes[0].set_title('PCA decomposition')
    axes[1].set_title('SVD decomposition')
    axes[2].set_title('TSNE decomposition')

    fig.savefig(f'./visualization/sampling_{epoch}.png')
    plt.close(fig)
