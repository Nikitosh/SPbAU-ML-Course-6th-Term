import numpy as np
import re
import csv
import math

FILE_PATH = "spam.txt"
RATIO = 0.8


# Returns dict with number of occurrences of each word in the text.
def get_words_count(text):
    words_count = {}
    for word in re.findall("[a-zA-Z]+", text):
        words_count[word] = words_count.get(word, 0) + 1
    return words_count


# Returns matrix with number of occurrences of each word in each text and
# dict with column number in this matrix for each word.
def vectorize(texts):
    vectors = []
    used_words = {}
    cur_column = 0
    for text in texts:
        words = get_words_count(text)
        for word in words.keys():
            if word not in used_words:
                used_words[word] = cur_column
                cur_column += 1
    for text in texts:
        vector = np.zeros(cur_column)
        words = get_words_count(text)
        for word, value in words.items():
            vector[used_words[word]] += value
        vectors.append(vector)
    return np.array(vectors), used_words


# Reads content of the file considering tab as delimiter between class labels and texts
# and returns these texts and class labels.
def read(file_path):
    reader = csv.reader(open(file_path, encoding='utf-8'), delimiter="\t")
    y = []
    texts = []
    for row in reader:
        y.append(row[0])
        texts.append(row[1])
    return texts, np.array(y)


class NaiveBayes:
    def __init__(self, alpha):
        self.alpha = alpha
        self.class_prior = {}
        self.word_count = []
        self.theta = []
        self.classes = None
        self.used_words = None

    # Builds Naive Bayes classifier for given texts and class labels.
    def fit(self, x, y):
        self.classes = np.unique(y)
        x, self.used_words = vectorize(x)
        for cur_class in self.classes:
            self.class_prior[cur_class] = np.mean(y == cur_class)
        for cur_class in self.classes:
            thetas = np.zeros(len(self.used_words))
            for i in range(len(y)):
                if y[i] == cur_class:
                    thetas += x[i]
            # Calculates thetas using additive smoothing.
            thetas = (thetas + self.alpha) / (thetas.sum() + self.alpha * len(self.used_words))
            self.theta.append(thetas)
            self.word_count.append(thetas.sum())
        self.theta = np.array(self.theta)

    # Classifies given texts using Naive Bayes classifier built in fit() method.
    def predict(self, x):
        y = []
        for text in x:
            words = get_words_count(text)
            max_a = -np.inf
            best_class = None
            # Chooses the class with maximal P(y) * \Pi theta_{yj}^N_j / N_j!,
            # where N_j is the number of occurrences of j-th word in text.
            for i in range(len(self.classes)):
                cur_class = self.classes[i]
                a = math.log(self.class_prior[cur_class])
                for word, value in words.items():
                    if word in self.used_words:
                        theta = self.theta[i][self.used_words[word]]
                    else:
                        theta = self.alpha / (self.word_count[i] + self.alpha * len(self.used_words))
                    a += value * math.log(theta) - sum([math.log(i) for i in range(1, value + 1)])
                if a > max_a:
                    max_a = a
                    best_class = cur_class
            y.append(best_class)
        return np.array(y)

    # Calculates the proportion of correct answers on test sample.
    def score(self, x, y):
        y_predict = self.predict(x)
        return np.mean(y_predict == y)


def main():
    texts, y = read(FILE_PATH)
    split_index = int(RATIO * len(texts))
    naive_bayes = NaiveBayes(alpha=1)
    naive_bayes.fit(texts[:split_index], y[:split_index])
    print(naive_bayes.score(texts[split_index:], y[split_index:]))

if __name__ == "__main__":
    main()
