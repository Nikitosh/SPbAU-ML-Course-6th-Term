import numpy as np
import matplotlib.pyplot as plt

PATH = 'pima-indians-diabetes.csv'
FEATURE_NUMBER = 9
RATIO = 0.8
ALPHAS = [1e-6, 1e-4, 1e-2, 1]
KS = [1, 10, 50]


# Reads data from given file and converts it to array of features and answers.
def read_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=FEATURE_NUMBER)
    return np.c_[-np.ones(len(data)), data[:, :-1]], \
           np.apply_along_axis(lambda x: np.where(x == 1, -1, 1), 0, data[:, -1])


# Splits data into two sets: for training and testing.
def train_test_split(x, y, ratio):
    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    train_length = round(len(x) * ratio)
    return x[:train_length], y[:train_length], x[train_length:], y[train_length:]


# Converts data in such manner that every feature has mean equals to 0 and variance equals to 1.
def standardize(data):
    cropped_data = data[:, 1:]
    return np.c_[data[:, 0], np.divide(cropped_data - np.mean(cropped_data, axis=0), np.std(cropped_data, axis=0))]


# Prints results of experiment (precision and recall).
def print_precision_recall(y_predicted, y_test):
    classes = set(y_test)
    for c in classes:
        tp = len(y_predicted[(y_predicted == c) & (y_test == c)])
        fp = len(y_predicted[(y_predicted == c) & (y_test != c)])
        fn = len(y_predicted[(y_predicted != y_test) & (y_predicted != c)])
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(c, precision, recall)


def log_loss(x):
    return np.log2(1 + np.exp(-x)), -1 / (np.log(2) * (1 + np.exp(x)))


def sigmoid_loss(x):
    return 2 / (1 + np.exp(x)), -2 * np.exp(x) / ((np.exp(x) + 1) ** 2)


def initialize_zero_weights(dimension):
    return np.zeros(dimension)


# Returns random vector of given dimension with norm equals to 1.
def initialize_random_weights(dimension):
    ans = np.random.rand(dimension) - 0.5
    ans /= np.linalg.norm(ans)
    return ans


# Return indent for given w.
def get_m(x, y, w):
    return np.multiply(x.dot(w), y)


class GradientDescent:
    def __init__(self, alpha, threshold=1e-6, loss=sigmoid_loss):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        if threshold <= 0:
            raise ValueError("threshold should be positive")
        self.alpha = alpha
        self.threshold = threshold
        self.loss = loss
        self.weights = None

    # Trains using given data no more than max_iter episodes (used for timeout if gradient descent works bad).
    def fit(self, x, y, max_iter=1e9):
        w = initialize_random_weights(x.shape[1])
        old_w = w + 2 * self.threshold # Adding is used for while condition failure.
        errors = []
        while np.linalg.norm(w - old_w) > self.threshold:
            l, l_derivative = self.loss(get_m(x, y, w))
            q = np.sum(l)
            grad_q = np.sum(np.multiply(np.multiply(np.transpose(x), y), l_derivative), axis=1)
            w, old_w = w - self.alpha * grad_q, w
            w /= np.linalg.norm(w)
            errors.append(q)
            max_iter -= 1
            if max_iter <= 0:
                break
        self.weights = w
        return errors

    def predict(self, x):
        return np.sign(x.dot(self.weights))


class SGD:
    def __init__(self, alpha, loss=log_loss, k=1, n_iter=100):
        if alpha <= 0:
            raise ValueError("alpha should be positive")
        if k <= 0 or not isinstance(k, int):
            raise ValueError("k should be a positive integer")
        if n_iter <= 0 or not isinstance(n_iter, int):
            raise ValueError("n_iter should be a positive integer")
        self.k = k
        self.n_iter = n_iter
        self.alpha = alpha
        self.loss = loss
        self.weights = None

    def fit(self, x, y):
        eta = 1 / len(x)
        self.weights = initialize_random_weights(x.shape[1])
        l, _ = self.loss(get_m(x, y, self.weights))
        q = np.sum(l)
        errors = []
        for i in range(self.n_iter):
            indices = np.random.choice(len(x), self.k)
            x0, y0 = x[indices], y[indices]
            l, l_derivative = self.loss(get_m(x0, y0, self.weights))
            eps = np.sum(l)
            self.weights -= self.alpha * np.sum(np.multiply(np.multiply(np.transpose(x0), y0), l_derivative), axis=1)
            self.weights /= np.linalg.norm(self.weights)
            q = (1 - eta) * q + eta * eps
            errors.append(q)
        return errors

    def predict(self, x):
        return np.sign(x.dot(self.weights))


def test_gradient_descent(x_train, y_train, x_test, y_test):
    gradient_descent = GradientDescent(alpha=1e-6)
    gradient_descent.fit(x_train, y_train)
    y_predicted = gradient_descent.predict(x_test)
    print_precision_recall(y_predicted, y_test)


def test_stochastic_gradient_descent(x_train, y_train, x_test, y_test):
    stochastic_gradient_descent = SGD(alpha=1e-4, n_iter=1000)
    stochastic_gradient_descent.fit(x_train, y_train)
    y_predicted = stochastic_gradient_descent.predict(x_test)
    print_precision_recall(y_predicted, y_test)


def build_gradient_descent_charts(x, y, alphas, losses):
    for i in range(len(losses)):
        loss = losses[i]
        for alpha in alphas:
            gradient_descent = GradientDescent(alpha=alpha, loss=loss)
            errors = gradient_descent.fit(x, y, max_iter=100)
            plt.figure(i)
            plt.plot(np.arange(1, len(errors) + 1), errors, label='alpha=' + str(alpha))
        plt.legend(title='loss=' + loss.__name__)
        plt.savefig('gradient_descent_' + loss.__name__ + '.png')
    plt.close()


def build_stochastic_gradient_descent_charts(x, y, alphas, ks, losses):
    for i in range(len(losses)):
        loss = losses[i]
        for j in range(len(ks)):
            for alpha in alphas:
                k = ks[j]
                stochastic_gradient_descent = SGD(alpha=alpha, k=k, loss=loss, n_iter=3000)
                errors = stochastic_gradient_descent.fit(x, y)
                plt.figure(i)
                plt.subplot(1, len(ks), j + 1)
                plt.plot(np.arange(1, len(errors) + 1), errors, label='alpha=' + str(alpha))
        plt.legend(title='loss=' + loss.__name__)
        plt.savefig('stochastic_gradient_descent_' + loss.__name__ + '.png')
    plt.close()


def main():
    np.random.seed(239)
    x, y = read_data(PATH)
    x = standardize(x)
    x_train, y_train, x_test, y_test = train_test_split(x, y, RATIO)
    test_gradient_descent(x_train, y_train, x_test, y_test)
    test_stochastic_gradient_descent(x_train, y_train, x_test, y_test)
    build_gradient_descent_charts(x_train, y_train, ALPHAS, [sigmoid_loss, log_loss])
    build_stochastic_gradient_descent_charts(x_train, y_train, ALPHAS, KS, [sigmoid_loss, log_loss])

if __name__ == "__main__":
    main()
