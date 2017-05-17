import numpy as np
import matplotlib.pyplot as plt

FEATURE_NUMBER = 14
FILE_PATH = 'boston.csv'
RATIO = 0.8


# Moves all values to [0, 1] segment.
def normalize(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


# Reads data from given file and converts it to array of features and answers.
def read_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=FEATURE_NUMBER + 1)
    data = normalize(data)
    return data[:, :-1], data[:, -1]


# Returns mean squared error.
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


# Implements regression predictor using solving normal equations system.
class NormalLR:
    def __init__(self, tau=0):
        self.tau = tau
        self.weights = None

    def fit(self, x, y):
        inversed_matrix = np.linalg.inv(np.dot(x.T, x) + self.tau * np.eye(x.shape[1]))
        self.weights = np.dot(np.dot(inversed_matrix, x.T), y)

    def predict(self, x):
        return np.dot(x, self.weights)


# Implements regression predictor using gradient descent.
class GradientLR(NormalLR):
    def __init__(self, alpha, tau=0):
        super().__init__(tau)
        if alpha <= 0:
            raise ValueError('alpha should be positive')
        self.alpha = alpha
        self.threshold = alpha / 100

    def fit(self, x, y):
        weights = np.random.randn(x.shape[1])
        while True:
            new_weights = weights - self.alpha * (np.dot((np.dot(x, weights) - y).T, x) / len(y) + self.tau)
            if np.linalg.norm(new_weights - weights) < self.threshold:
                break
            weights = new_weights
        self.weights = weights


def sample(size, weights):
    x = np.ones((size, 2))
    x[:, 1] = np.random.gamma(4., 2., size)
    y = x.dot(np.asarray(weights))
    y += np.random.normal(0, 1, size)
    return x[:, 1:], y


def visualize(lr, size):
    x, y_true = sample(size, weights=[24., 42.])
    lr.fit(x, y_true)
    plt.scatter(x, y_true)
    plt.plot(x, lr.predict(x), color='red')
    plt.show()


# Returns mean squared error for predicting on x_test after fitting on x_train and y_train.
def get_error(x_train, y_train, x_test, y_test, lr):
    lr.fit(x_train, y_train)
    y_predicted = lr.predict(x_test)
    return mse(y_test, y_predicted)


# Splits data into two sets: for training and testing.
def train_test_split(x, y, ratio):
    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    train_length = round(len(x) * ratio)
    return x[:train_length], y[:train_length], x[train_length:], y[train_length:]


# Tests two linear regression predictors on artificial sample.
def test_sample(normal_lr, gradient_lr):
    sample_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    errors_normal_lr = []
    errors_gradient_lr = []
    for sample_size in sample_sizes:
        x, y = sample(sample_size, weights=[24., 42.])
        x = normalize(x)
        y = normalize(y)
        x_train, y_train, x_test, y_test = train_test_split(x, y, RATIO)
        errors_normal_lr.append(get_error(x_train, y_train, x_test, y_test, normal_lr))
        errors_gradient_lr.append(get_error(x_train, y_train, x_test, y_test, gradient_lr))
    return errors_normal_lr, errors_gradient_lr


# Tests two linear regression predictors on given data.
def test_data(normal_lr, gradient_lr):
    x, y = read_data(FILE_PATH)
    x_train, y_train, x_test, y_test = train_test_split(x, y, RATIO)
    return get_error(x_train, y_train, x_test, y_test, normal_lr), \
           get_error(x_train, y_train, x_test, y_test, gradient_lr)


def main():
    np.random.seed(239)

    errors_normal_lr, errors_gradient_lr = test_sample(NormalLR(), GradientLR(0.1))
    print(errors_normal_lr)
    print(errors_gradient_lr)

    error_normal_lr, error_gradient_lr = test_data(NormalLR(), GradientLR(0.1))
    print(error_normal_lr)
    print(error_gradient_lr)


if __name__ == "__main__":
    main()
