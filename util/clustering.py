"""
Clustering functionality 
Code by Lukas
"""
from tqdm.notebook import tqdm
import matplotlib.ticker as mtick
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
import pandas as pd
import numpy as np
import util.plotting as pl
import os

def show_images_row(fpths, title=None):
    # title: either string or list of the size of fpths
    fig = figure()
    number_of_files = len(fpths)
    for i in range(number_of_files):
        ax = fig.add_subplot(1, number_of_files, i + 1)
        image = imread(fpths[i])
        imshow(image, cmap='Greys_r')
        if title is not None:
            if isinstance(title, str) and i == 0:
                plt.title(title, loc="Left")
            elif len(title) == len(fpths):
                plt.title(title[i], fontsize=15)
        axis('off')
    return fig

def get_clustering_dict(labels):
    """
    Transform labels from list to dict of {cluster index: [image index, image index, ...]}
    :param labels: numpy array of shape (num_images,2). columns: [image index, cluster index]
    :return: dict
    """
    clusters = {}
    labels_sorted = pd.Series(labels).value_counts().sort_values(ascending=False)
    for i in list(labels_sorted.index):
        clusters[i] = np.array(np.where(labels == i)).reshape(-1)
    return clusters


def get_centroid_idx(idxs, features, top_n=5, distance_metric="cosine"):
    """
    Get most central images
    :param idxs: list of image indices corresponding to the feature array
    :param top_n: get n most central images
    :param features: array of feature vectors.
    :param distance_metric: cosine/euclidean metric for distance comparisons
    :return: list of indices
    """
    features_cluster = features[idxs]
    # find most central point (=center) in cluster
    avg_point = np.mean(features_cluster, axis=0)

    # get distance of each image in cluster to the center
    if distance_metric == "cosine":
        distances = np.dot(features_cluster, avg_point)
    else:
        distances = np.linalg.norm(features_cluster - avg_point, axis=1)

    # get top n most central images based on distance to center
    indices_cosine_sorted = np.argsort(distances)  # returns indices sorted by ascending sort values
    if distance_metric == "cosine":
        # relative = indices are w.r.t to the images in cluster, not all images
        idxs_relative = indices_cosine_sorted[::-1][:top_n]
    else:
        idxs_relative = indices_cosine_sorted[::1][:top_n]

    return np.array(idxs)[idxs_relative]


def viz_clusters(clusters, image_db, image_db_folder, show_n_per_cluster=10, show_pics=True, show_centroids="one",
                 features=None):
    """
    Visualizes a given clustering by showing example images for each cluster
    :param clusters: dictionary of {cluster identifier: list of indices}
    :param image_db: list of image filenames
    :param image_db_folder: path to image dataset
    :param show_n_per_cluster: show n example images per cluster
    :param show_pics: if False, only print cluster sizes and don't show example images
    :param show_centroids: 'all' (all images are centroids), 'none' (all images are random), 'one' (first image is centroid, others are random)
    """
    assert show_centroids in ["all", "none", "one"]
    if show_centroids != 'none': assert features is not None, "If you want to see the centroids please provide the features used for the clustering"

    for cluster_id, idxs in clusters.items():
        sample_size = min(show_n_per_cluster, len(idxs))
        if show_centroids == "all":
            idcs_sample = get_centroid_idx(idxs, features, show_n_per_cluster)
        elif show_centroids == "one":
            idcs_sample = np.append(get_centroid_idx(idxs, features, 1),
                                    np.random.choice(idxs, sample_size-1, replace=False))
        else:
            idcs_sample = np.random.choice(idxs, sample_size, replace=False)
        if show_pics:
            if len(idxs) > 0:
                print("cluster {}  size: {} {} ({:.1%})".format(cluster_id, len(idxs), "|" * int(np.log1p(len(idxs))),
                                                                len(idxs) / len(image_db)))
                fig, axs = plt.subplots(1, max(2, sample_size), figsize=(30, 30))
                for ii, idx in enumerate(idcs_sample):
                    img = plt.imread(os.path.join(image_db_folder,image_db[idx]))
                    axs[ii].axis('off')
                    axs[ii].imshow(img)
                plt.show()
        else:
            print(cluster_id, len(idxs))

def viz_clusters_compact(clusters, image_db, image_db_folder,features, do_print=False):
    """
    Shows a compact clustering visualization with 1 centroid image per cluster
    :param clusters: dictionary of {cluster identifier: list of indices}
    :param image_db: list of image names
    :param image_db_folder: path to the image folder
    :param features: features used for the clustering
    :param do_print: print fpaths of images to plot
    """
    images, ratios = [], []

    for cluster_name, idxs in clusters.items():
        # get the centroid image
        img_idx = get_centroid_idx(idxs, features, 1)[0] # returns a list
        images.append(image_db[img_idx])
        ratios.append("{}:{:.1%}".format(cluster_name,len(idxs)/len(image_db)))

    fpaths = [os.path.join(image_db_folder, img) for img in images]
    if do_print:
        print(os.path.abspath(image_db_folder), *images)
    return show_images_row(fpaths, title=ratios)

def perform_clustering(features, n_clusters, pca_components=20):
    """
    Convenience function for creating and fitting a k-nn clustering model
    :param features: np array of shape NxM
    :param n_clusters: number of clusters
    :param pca_components: number of PCA components
    :return: tuple of:
    - clustering: fitted clustering object
    - labels: cluster labels (numpy array of shape N)
    - labels_sorted: cluster labels sorted by occurrence (numpy array of shape N)
    - clusters: dict of {cluster:[list of indices]} (index referring to features)
    - features_pca: dimensionality-reduces features used for the clustering
    """
    if pca_components:
        # PCA is recommended for Kmeans (https://scikit-learn.org/stable/modules/clustering.html). PCA can introduce randomness so we set the random state
        features_pca = PCA(n_components=pca_components, random_state=42).fit_transform(features)
    else:
        features_pca = features

    clustering = KMeans(n_clusters=n_clusters, random_state=42)
    clustering.fit(features_pca)
    labels = clustering.labels_
    labels_sorted = pd.Series(labels).value_counts().sort_values(ascending=False)
    clusters = get_clustering_dict(labels)
    return clustering, labels, labels_sorted, clusters, features_pca


def cluster_and_plot(resources, n_clusters, cluster_names=None, areaplot_freq="Q"):
    """
    Convenience function for clustering an image dataset and creating a bunch of plots
    :param resources (dict): all necessary resources for clustering
    :param n_clusters: number of clusters
    :param cluster_names (dict): Manual naming of what the clusters represent. Set this to None on the first run when the clusters are still unknown.
    :param areaplot_freq (str): bin parameter for the cluster size over time areaplot ('Y':year,'Q':quarter,'W':week)
    """
    df, features, imagedb, image_folder = resources["df"], resources["features"], resources["image_db"], resources[
        "image_folder"]

    # perform clustering and get labels
    clustering, labels, labels_sorted, clusters, features_pca = perform_clustering(features, n_clusters=n_clusters)

    # visualize clusters by their images
    viz_clusters(clusters, imagedb, image_folder, show_n_per_cluster=20, show_pics=True, show_centroids=True,
                 features=features_pca)

    # write  clustering labels to df in new column
    if not "cluster" in df.columns:
        for i, cluster_i in enumerate(labels):
            df.loc[df["image"] == imagedb[i], "cluster"] = cluster_names[int(cluster_i)] if cluster_names else int(
                cluster_i)

    print("Overall cluster frequencies:\n", (df["cluster"].value_counts() / len(df)).astype(float).map("{:.1%}".format))

    # visualize the clustering labels over time
    pl.catplot_area(df, groupby="cluster", id_variable="image", freq=areaplot_freq, relative_to_group=True,
                    do_print=True)
    plt.show()

    grp = df.groupby([df.index.year, "cluster"])["image"].count().reset_index()
    data = grp.pivot(index="timestamp", columns="cluster", values="image")
    pl.stacked_barchart(data)
    plt.show()


def plot_silhouette_score(features, Algo, min_n=2, max_n=10, **kwargs):
    """
    Calculate and plot silhouette score for a model across different numbers of clusters
    :param features: numpy array of features
    :param Algo: sklearn clustering model (has to take the number of clusters as a parameter)
    :param min_n: minimum n to consider
    :param max_n: maximum n to consider
    :param kwargs: anything else you wish to pass to the model
    """
    x = []
    scores = []
    for n in tqdm(range(min_n, max_n + 1)):
        clustering = Algo(n)
        clustering.fit(features)
        labels = clustering.labels_
        x.append(n)
        scores.append(silhouette_score(features, labels))
    fig = plt.figure()
    plt.plot(x, scores, label="silhouette score", linewidth=2)
    plt.xlabel("n clusters")
    fig.legend(loc="upper right")
    plt.title(Algo.__name__)