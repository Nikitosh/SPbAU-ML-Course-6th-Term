import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from PIL import Image


def scale_to_unit_interval(ndar, eps=1e-8):
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(x, img_shape, tile_shape, tile_spacing=(0, 0), scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]
    h, w = img_shape
    hs, ws = tile_spacing

    dt = x.dtype
    if output_pixel_vals:
        dt = 'uint8'
    out_array = np.zeros(out_shape, dtype=dt)

    for tile_row in range(tile_shape[0]):
        for tile_col in range(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < x.shape[0]:
                this_x = x[tile_row * tile_shape[1] + tile_col]
                if scale_rows_to_unit_interval:
                    this_img = scale_to_unit_interval(this_x.reshape(img_shape))
                else:
                    this_img = this_x.reshape(img_shape)
                c = 1
                if output_pixel_vals:
                    c = 255
                out_array[tile_row * (h + hs): tile_row * (h + hs) + h, tile_col * (w + ws): tile_col * (w + ws) + w] \
                    = this_img * c
    return out_array


def visualize_mnist(x_train):
    images = x_train[0:2500, :]
    image_data = tile_raster_images(images,
                                    img_shape=[28,28],
                                    tile_shape=[50,50],
                                    tile_spacing=(2,2))
    im_new = Image.fromarray(np.uint8(image_data))
    im_new.save('mnist.png')


# Returns pair of sigmoid function and it's derivative.
def sigmoid(x):
    return 1 / (1 + np.exp(-x)), np.exp(x) / ((np.exp(x) + 1) ** 2)


class NeuralNetwork:
    def __init__(self, layers, activation=sigmoid):
        self.num_layers = len(layers)
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.u = []

    # Returns result of applying neural network with current weights on given vector x.
    def apply(self, x):
        for i in range(self.num_layers - 1):
            x = self.activation(np.dot(x, self.weights[i][:-1]) - self.weights[i][-1])[0]
        return x

    def train(self, x, y, max_iter=10000, alpha=1):
        self.weights = []
        for i in range(self.num_layers - 1):
            self.weights.append(np.random.rand(self.layers[i] + 1, self.layers[i + 1]) - 0.5)

        eta = 1 / len(x)
        # Converts labels into vectors with 1 value on position of label.
        labels = np.zeros((len(y), self.layers[-1]))
        for i in range(len(y)):
            labels[i][y[i]] = 1

        q = np.sum(np.square(self.apply(x) - labels)) / 2
        for j in range(max_iter):
            i = np.random.randint(0, len(x))
            self.forward(x[i])
            q = (1 - eta) * q + eta * self.backward(x[i], labels[i], alpha)

    def forward(self, x):
        self.u = [x]
        for i in range(self.num_layers - 1):
            x = self.activation(np.dot(x, self.weights[i][:-1]) - self.weights[i][-1])[0]
            self.u.append(x)

    def backward(self, x, y, alpha):
        eps = [self.u[-1] - y]
        for i in range(self.num_layers - 1):
            w = self.weights[-i - 1]
            sigma = self.activation(np.dot(self.u[-i - 2], w[:-1]) - w[-1])[1]
            eps.append(np.dot(w, np.multiply(eps[i], sigma))[:-1])
            self.weights[-i - 1] -= alpha * np.outer(np.append(self.u[-i - 2], -1), np.multiply(eps[i], sigma))
        # Returns new error.
        return np.sum(np.square(eps[0])) / 2

    def predict(self, x):
        return np.argmax(self.apply(x), axis=1)

    def score(self, x, y):
        y_predicted = self.predict(x)
        return np.mean(y_predicted == y)


def main():
    np.random.seed(239)
    dataset = datasets.fetch_mldata('MNIST Original')
    x_train, x_test, y_train, y_test = train_test_split(dataset.data / 255.0, dataset.target.astype("int0"),
                                                        test_size=0.3)
    visualize_mnist(x_train)
    n = x_train.shape[1]
    layers_list = [
        [n, 10],
        [n, 5, 10],
        [n, 25, 10],
        [n, 100, 10],
        [n, 10, 10, 10],
        [n, 25, 25, 10],
        [n, 50, 20, 10]
    ]
    for layers in layers_list:
        neural_network = NeuralNetwork(layers)
        neural_network.train(x_train, y_train, max_iter=2*len(x_train), alpha=0.01)
        print(layers, neural_network.score(x_test, y_test))

if __name__ == "__main__":
    main()
