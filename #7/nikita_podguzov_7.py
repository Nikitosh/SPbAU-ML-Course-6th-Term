import numpy as np
from cvxopt import spmatrix, matrix, solvers
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.mlab import find

EPS = 1e-5


# Visualizes given points using fitted SVM.
def visualize(clf, x, y):
    border = .5
    h = .02

    x_min, x_max = x[:, 0].min() - border, x[:, 0].max() + border
    y_min, y_max = x[:, 1].min() - border, x[:, 1].max() + border

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]

    z_class = clf.predict(mesh).reshape(xx.shape)

    # Put the result into a color plot
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx, yy, z_class, cmap=plt.cm.summer, alpha=0.3)

    # Plot hyperplane and margin
    z_dist = clf.decision_function(mesh).reshape(xx.shape)
    plt.contour(xx, yy, z_dist, [0.0], colors='black')
    plt.contour(xx, yy, z_dist, [-1.0, 1.0], colors='black', linestyles='dashed')

    # Plot also the training points
    y_predicted = clf.predict(x)

    ind_support = clf.support
    ind_correct = list(set(find(y == y_predicted)) - set(ind_support))
    ind_incorrect = list(set(find(y != y_predicted)) - set(ind_support))

    plt.scatter(x[ind_correct, 0], x[ind_correct, 1], c=y[ind_correct], cmap=plt.cm.summer, alpha=0.9)
    plt.scatter(x[ind_incorrect, 0], x[ind_incorrect, 1], c=y[ind_incorrect], cmap=plt.cm.summer, alpha=0.9, marker='*',
                s=50)
    plt.scatter(x[ind_support, 0], x[ind_support, 1], c=y[ind_support], cmap=plt.cm.summer, alpha=0.9, linewidths=1.8,
                s=40)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Implements linear SVM.
class LinearSVM:
    def __init__(self, c):
        self.c = c
        self.w0 = 0
        self.w = None
        self.support = None

    def fit(self, x, y):
        l, n = x.shape
        p = spmatrix(1, range(n), range(n), size=(n + l + 1, n + l + 1))
        q = matrix(spmatrix(self.c, range(n + 1, n + l + 1), [0] * l))
        g1 = matrix(np.zeros((l, n + 1)))
        g2 = g4 = matrix(-np.eye(l))
        g3 = matrix(np.c_[np.transpose(np.multiply(np.transpose(x), -y)), -y])
        g = matrix([[g1, g3], [g2, g4]])
        h = matrix(spmatrix(-1, range(l, 2 * l), [0] * l))

        solution = solvers.qp(p, q, g, h)
        self.w = np.matrix.flatten(np.array(solution['x'])[:n])
        self.w0 = np.array(solution['x'])[n]
        xi = np.ndarray.flatten(np.array(solution['x'][n + 1:]))
        self.support = find(np.abs(np.multiply(y, np.dot(x, self.w) + self.w0) - (1 - xi)) < EPS)

    def decision_function(self, x):
        return np.dot(x, self.w) + self.w0

    def predict(self, x):
        return np.sign(self.decision_function(x))


def linear_kernel():
    return lambda x1, x2: np.dot(x1, x2)


def homogeneous_polynomial_kernel(degree):
    return lambda x1, x2: np.dot(x1, x2) ** degree


def polynomial_kernel(degree):
    return lambda x1, x2: (np.dot(x1, x2) + 1) ** degree


def gaussian_kernel(sigma):
    return lambda x1, x2: np.exp(-sigma * np.linalg.norm(x1 - x2, axis=1) ** 2)


# Implements kernel SVM.
class KernelSVM:
    def __init__(self, c, kernel=None, sigma=1.0, degree=2):
        self.c = c
        kernels = {
            linear_kernel: linear_kernel(),
            homogeneous_polynomial_kernel: homogeneous_polynomial_kernel(degree),
            polynomial_kernel: polynomial_kernel(degree),
            gaussian_kernel: gaussian_kernel(sigma)
        }
        self.kernel = kernels[kernel]
        self.w0 = 0
        self.x = None
        self.y = None
        self.alpha = None
        self.support = None

    def fit(self, x, y):
        self.x = x
        self.y = y
        l, n = x.shape
        p = matrix([(y * y[j] * self.kernel(x, x[j])).tolist() for j in range(l)])
        q = matrix(-np.ones(l))
        g = matrix([[matrix(np.eye(l)), matrix(-np.eye(l))]])
        h = matrix(spmatrix(self.c, range(l), [0] * l, size=(2 * l, 1)))
        a = matrix(y * 1.).trans()
        b = matrix([0.])
        solution = solvers.qp(p, q, g, h, a, b)
        self.alpha = np.array(solution['x'])
        self.support = find(self.alpha > EPS)
        self.w0 = np.mean(y[self.support] -
                          sum([self.alpha[j] * y[j] * self.kernel(x[self.support], x[j]) for j in range(l)]))

    def decision_function(self, x):
        res = self.w0 * np.ones(len(x))
        for i in self.support:
            res += self.alpha[i] * self.y[i] * self.kernel(x, self.x[i])
        return res

    def predict(self, x):
        return np.sign(self.decision_function(x))


# Tests given SVM on sample and saves visualization to filename.
def test(svm, x, y, filename):
    svm.fit(x, y)
    visualize(svm, x, y)
    plt.savefig(filename)
    plt.close()


# Tests different SVMs on sample and saves visualization to filename.
def test_svms(x, y):
    test(LinearSVM(1), x, y, 'Linear SVM.png')
    test(KernelSVM(1, linear_kernel), x, y, 'Kernel SVM (linear).png')
    degrees = [2, 3, 4]
    for degree in degrees:
        test(KernelSVM(1, polynomial_kernel, degree=degree), x, y, 'Kernel SVM (polynomial)' + str(degree) + '.png')
    for degree in degrees:
        test(KernelSVM(1, homogeneous_polynomial_kernel, degree=degree), x, y,
             'Kernel SVM (homogeneous polynomial)' + str(degree) + '.png')
    sigmas = [0.1, 0.25, 0.5, 1, 1.5, 2, 5]
    for sigma in sigmas:
        test(KernelSVM(1, gaussian_kernel, sigma=sigma), x, y, 'Kernel SVM (gaussian)' + str(sigma) + '.png')


def main():
    np.random.seed(239)
    solvers.options['show_progress'] = False
    x, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=3)
    y = 2 * y - 1 # We want to get -1/1 instead of 0/1 for y.
    test_svms(x, y)

if __name__ == "__main__":
    main()
