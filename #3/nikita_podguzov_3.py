import numpy as np
import pandas as pd
import math
import numbers
from PIL import Image, ImageDraw

SHIFT = 100
TYPE = 'type'
COLORS = {'Goblin': (0, 255, 0), 'Ghoul': (0, 0, 0), 'Ghost': (0, 0, 0)}
RATIO = 0.8


class Predicate:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    # Returns string representation of predicate.
    def get_text_representation(self):
        if isinstance(self.value, numbers.Number):
            sign = '<='
        else:
            sign = '=='
        return self.column + ' ' + sign + ' ' + str(self.value)

    # Returns if predicate is satisfied by given data (one row).
    def is_satisfied(self, data):
        t, f = self.divide(data)
        return not t.empty

    # Returns given data divided by condition of predicate satisfying.
    def divide(self, data):
        if isinstance(self.value, numbers.Number):
            return data[data[self.column] <= self.value], data[data[self.column] > self.value]
        else:
            return data[data[self.column] == self.value], data[data[self.column] != self.value]

    # Returns classification divided by condition of predicate satisfying.
    def divide_y(self, data, y):
        if isinstance(self.value, numbers.Number):
            return y[data[self.column] <= self.value], y[data[self.column] > self.value]
        else:
            return y[data[self.column] == self.value], y[data[self.column] != self.value]


class Node:
    def __init__(self, predicate=None, result=None, false_branch=None, true_branch=None):
        self.predicate = predicate
        self.result = result
        self.false_branch = false_branch
        self.true_branch = true_branch


# Calculates entropy of given data.
def entropy(y):
    length = len(y)
    return y.value_counts().apply(lambda x: -x / length * math.log(x / length, 2)).sum()


# Calculates width of subtree with given root.
def get_width(node):
    if node is None:
        return 0
    return math.ceil(max(0.5, get_width(node.false_branch)) + max(0.5, get_width(node.true_branch)))


# Calculates height of subtree with given root.
def get_depth(node):
    if node is None:
        return 0
    return 1 + max(get_depth(node.false_branch), get_depth(node.true_branch))


# Returns the best predicate for ID3 with given score function.
def get_best_predicate(xs, y, score):
    best_information_gain = 0
    best_predicate = None
    for column in list(xs):
        values = xs[column].value_counts().index.tolist()
        for value in values:
            predicate = Predicate(column, value)
            t_y, f_y = predicate.divide_y(xs, y)
            information_gain = score(y) - len(t_y) / len(y) * score(t_y) - len(f_y) / len(y) * score(f_y)
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_predicate = predicate
    return best_predicate


class DecisionTree:
    def __init__(self, root=None):
        self.root = root

    # Builds decision tree with ID3 algorithm.
    def build(self, xs, y, score=entropy):
        if len(y.unique()) == 1:  # If all objects are in the same class.
            self.root = Node(result=y.iloc[0])
        else:
            predicate = get_best_predicate(xs, y, score)
            t, f = predicate.divide(xs)
            t_y, f_y = predicate.divide_y(xs, y)
            if t.empty or f.empty:
                self.root = Node(result=y.value_counts().idxmax())  # Return Node with majority(y)
            else:
                l = DecisionTree().build(f, f_y, score).root
                r = DecisionTree().build(t, t_y, score).root
                self.root = Node(predicate=predicate, false_branch=l, true_branch=r)
        return self

    # Predicts class for given x (can be used only after decision tree is built).
    def predict(self, x):
        node = self.root
        while node.result is None:
            if node.predicate.is_satisfied(x):
                node = node.true_branch
            else:
                node = node.false_branch
        return node.result


# Draws tree.
def draw_tree(tree, path='tree.jpg'):
    w = get_width(tree) * SHIFT
    h = get_depth(tree) * SHIFT
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_node(draw, tree, w / 2, 20)
    img.save(path, 'JPEG')


# Draws subtree with given root in (x, y).
def draw_node(draw, tree, x, y):
    if tree.result is None:
        width1 = get_width(tree.false_branch) * SHIFT
        width2 = get_width(tree.true_branch) * SHIFT
        left = x - (width1 + width2) / 2
        right = x + (width1 + width2) / 2
        predicate_text = tree.predicate.get_text_representation()
        draw.text((x - 20, y - 10), predicate_text, (0, 0, 0))
        draw.line((x, y, left + width1 / 2, y + SHIFT), fill=(255, 0, 0))
        draw.line((x, y, right - width2 / 2, y + SHIFT), fill=(255, 0, 0))
        draw_node(draw, tree.false_branch, left + width1 / 2, y + SHIFT)
        draw_node(draw, tree.true_branch, right - width2 / 2, y + SHIFT)
    else:
        draw.text((x - 20, y), tree.result, COLORS[tree.result])


# Splits column with given name from data frame.
def split_by_name(data_frame, name):
    return data_frame[name], data_frame.drop(name, axis=1)


# Splits data into two sets: for training and testing.
def train_test_split(x, y, ratio):
    xy = pd.concat([x, y], axis=1)
    xy = xy.reindex(np.random.permutation(xy.index))
    y, x = split_by_name(xy, TYPE)
    train_length = round(len(x) * ratio)
    return x[:train_length], y[:train_length], x[train_length:], y[train_length:]


# Returns proportion of errors. Can be used for testing.
def test(x, y):
    x_train, y_train, x_test, y_test = train_test_split(x, y, RATIO)
    tree = DecisionTree().build(x_train, y_train)
    errors = 0
    for i in x_test.index:
        if y_test[i] != tree.predict(x_test.loc[[i]]):
            errors += 1
    return errors / len(y_test)


def main():
    data = pd.read_csv('halloween.csv')
    types, data = split_by_name(data, TYPE)
    # print('errors: ' + str(test(data, types)))
    tree = DecisionTree().build(data, types)
    draw_tree(tree.root)

if __name__ == "__main__":
    main()
