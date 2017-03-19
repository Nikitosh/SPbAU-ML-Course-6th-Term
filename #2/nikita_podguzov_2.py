import numpy as np
import cv2
import random
import sys
import os
from itertools import groupby

# Number of colors.
N_COLORS = 16
# Width of bar chart with color distribution.
BAR_WIDTH = 400
# Height of bar chart with color distribution.
BAR_HEIGHT = 50


# Returns p-norm distance between point and list of points with p = 2.
def distance2(p, points):
    return np.linalg.norm(p - points, axis=1)


# Reads image [width, height] with given path and returns matrix [width*height, 3] with BGR components.
def read_image(path):
    image = cv2.imread(path)
    shape = image.shape
    image = image.reshape(-1, image.shape[-1])
    return image, shape


# Writes image with given path and shape.
def write_image(path, image, shape):
    image = image.reshape(shape)
    cv2.imwrite(path, image)


# Appends suffix to file name ignoring it's extension.
def append_suffix(filename, suffix):
    name, extension = os.path.splitext(filename)
    return "{name}_{suffix}{extension}".format(name=name, suffix=suffix, extension=extension)


# Returns mean centroids for given centroids distribution.
def get_mean_centroids(x, closest_centroids, n_clusters):
    return np.array([x[closest_centroids == i].mean(axis=0) for i in range(n_clusters)])


# Returns list of nearest centroids for given list of points.
def get_clusters(x, centroids, distance_metric):
    return np.array([distance_metric(centroid, x) for centroid in centroids]).argmin(axis=0)


# Returns centroids initialization as n_clusters-sample from x
def forgy_initialization(x, n_clusters, distance_metric):
    return x[np.random.choice(len(x), n_clusters)]


# Returns random centroids initialization
def random_initialization(x, n_clusters, distance_metric):
    random_clusters = x.apply_along_axis(random.randrange(n_clusters), axis=1)
    return get_mean_centroids(x, random_clusters, n_clusters)


# Returns initialization which is used in k-means++ method.
def k_means_plus_plus_initialization(x, n_clusters, distance_metric):
    centroids = [x[np.random.choice(len(x))]]
    for i in range(n_clusters - 1):
        distances = np.array([distance_metric(centroid, x) for centroid in centroids]).min(axis=0)
        centroids.append(x[np.random.choice(len(x), p=distances / distances.sum())])
    return np.array(centroids)


def k_means(x, n_clusters, distance_metric):
    return k_means_common(x, n_clusters, distance_metric, forgy_initialization)


def k_means_plus_plus(x, n_clusters, distance_metric):
    return k_means_common(x, n_clusters, distance_metric, k_means_plus_plus_initialization)


def k_means_common(x, n_clusters, distance_metric, initialize):
    means = initialize(x, n_clusters, distance_metric)
    while True:
        old_means = means
        clusters = get_clusters(x, means, distance_metric)
        means = get_mean_centroids(x, clusters, n_clusters)
        if np.array_equal(old_means, means):
            break
    return get_clusters(x, means, distance_metric), means


# Creates histogram with colors distribution.
def centroid_histogram(labels):
    return np.array([len(list(group)) / len(labels) for key, group in groupby(sorted(labels))])


# Creates bar chart for given histogram.
def plot_colors(hist, centroids):
    bar = np.zeros((BAR_HEIGHT, BAR_WIDTH, 3), np.uint8)
    start_x = 0
    for (percent, color) in zip(hist, centroids):
        end_x = start_x + percent * BAR_WIDTH
        cv2.rectangle(bar, (int(start_x), 0), (int(end_x), BAR_HEIGHT), color.astype("uint8").tolist(), -1)
        start_x = end_x
    return bar


# Recolors given image with n_colors colors. Can use either k_means_plus_plus or k_means method.
def recolor(image, n_colors):
    clusters, centroids = k_means_plus_plus(image, n_colors, distance2)
    # clusters, centroids = k_means(image, n_colors, distance2)
    histogram = centroid_histogram(clusters)
    bar = plot_colors(histogram, centroids)
    image = np.array(list(map(lambda x: centroids[clusters[x]], range(len(image)))))
    return image, bar


# First arguments stands for path of image.
def main():
    path = sys.argv[1]
    image, shape = read_image(path)
    image_n_colors, bar = recolor(image, N_COLORS)
    write_image(append_suffix(path, str(N_COLORS) + 'colors'), image_n_colors, shape)
    cv2.imwrite(append_suffix(path, 'bar_chart'), bar)

if __name__ == "__main__":
    main()
