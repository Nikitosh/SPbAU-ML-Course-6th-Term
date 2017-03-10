import numpy as np
import pandas as pd
import random
from scipy.stats import mode

CLASS = "Class 1/2/3"
RATIO = 0.6


# Splits column with given name from data frame.
def split_by_name(data_frame, name):
    return data_frame[name], data_frame.drop(name, axis=1)


# Moves all values to [0, 1] segment.
def normalize(data_frame):
    return (data_frame - data_frame.min()) / (data_frame.max() - data_frame.min())


# Splits data into two sets: for training and testing.
def train_test_split(x, y, ratio):
    xy = pd.concat([x, y], axis=1)
    xy = xy.reindex(np.random.permutation(xy.index))
    y, x = split_by_name(xy, CLASS)
    train_length = round(len(x) * ratio)
    return x[:train_length], y[:train_length], x[train_length:], y[train_length:]


# p-norm distance with p = 1.
def distance1(p1, p2):
    return np.linalg.norm(p1 - p2, ord=1)


# p-norm distance with p = 2.
def distance2(p1, p2):
    return np.linalg.norm(p1 - p2)


# Finds the most frequent class against nearest k points for given x.
def find_nearest(x_train, y_train, x, k, dist):
    distances = x_train.apply(lambda z: dist(x, z), axis=1)
    return mode(y_train.iloc[distances.values.argsort()][:k])[0][0]


# Method of k nearest neighbours.
def knn(x_train, y_train, x_test, k, dist):
    res = x_test.apply(lambda x: find_nearest(x_train, y_train, x, k, dist), axis=1)
    return res


# Prints results of experiment (precision and recall).
def print_precision_recall(y_pred, y_test):
    classes = set(y_test)
    for c in classes:
        tp = len(y_pred[(y_pred == c) & (y_test == c)])
        fp = len(y_pred[(y_pred == c) & (y_test != c)])
        fn = len(y_pred[(y_pred != y_test) & (y_pred != c)])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(c, precision, recall)


# Returns number of bad predictions for given k.
def loo(x_train, y_train, k, dist):
    bad_pred = 0
    for i in range(len(x_train.index)):
        if (y_train.iloc[i] != knn(x_train.drop(x_train.index[[i]]), y_train.drop(y_train.index[[i]]),
                                   x_train.iloc[[i]], k, dist).iloc[0]):
            bad_pred += 1
    return bad_pred


# Searches optimal k with Leave-One-Out method.
def loocv(x_train, y_train, dist):
    best_pred = len(x_train.index) + 1
    k_opt = -1
    # After some experiments we can see that optimal k is always relative small,
    # so we bound it by length of training set divided by 3 for faster working.
    for k in range(1, len(x_train.index) // 3 + 1):
        cur_pred = loo(x_train, y_train, k, dist)
        if cur_pred < best_pred:
            best_pred = cur_pred
            k_opt = k
    return k_opt


def main():
    random.seed(239)

    data = pd.read_csv('wine.csv')
    classes, data = split_by_name(data, CLASS)
    data = normalize(data)
    x_train, y_train, x_test, y_test = train_test_split(data, classes, RATIO)

    # We consider two distances: p-norm distances with p = 1 and p = 2.
    for distance in [distance1, distance2]:
        print(distance.__name__)
        k_opt = loocv(x_train, y_train, distance)
        print("k: " + str(k_opt))
        y_pred = knn(x_train, y_train, x_test, k_opt, distance)
        print_precision_recall(y_pred, y_test)

    # As we see, results (precision and recall for every class) are very close to 1
    # (specifically, all precisions > 0.95).
    # It shows that such simple method works very good in this problem, so we refuted the myth,
    # that all wines taste the same.

if __name__ == "__main__":
    main()

