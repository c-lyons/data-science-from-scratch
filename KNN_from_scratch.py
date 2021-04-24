from sortedcontainers import SortedList
import numpy as np
import pandas as pd
from datetime import datetime

def get_data(limit=None):
    print("Reading image data and transforming...")
    df = pd.read_csv('data/mnist_sample.csv')
    data = df.to_numpy()
    np.random.shuffle(data)
    X = data[:, 1:] / 255 # normalising pixel values from 0->1
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y

class KNN(object):
    def __init__ (self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))  # initializing y column as zeros array of size X
        for i, x in enumerate(X):
            sl = SortedList()  # predefining sorted list to size K
            for j, x_train in enumerate(self.X):
                diff = x - x_train
                d = diff.dot(diff)  # squared difference using dot
                if len(sl) < self.k:  # if less than K neighbours, add to list
                    sl.add((d, self.y[j]))
                elif d < sl[-1][0]:
                    del sl[-1]
                    sl.add((d, self.y[j]))
            votes = {}  # empty dict for votes
            for k, v in sl:
                votes[v] = votes.get(v, 0) + 1 # counting votes for v in sortedlist
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class
        return y

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P==y)  # return mean of prediction == true label

if __name__ == '__main__':
    X, y = get_data(limit=5000)
    n_train = 4000
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    for k in range(1, 10):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(X_train, y_train)
        print('Training time: ', (datetime.now()-t0))
        t0 = datetime.now()
        print('Training Accuracy: ', knn.score(X_train, y_train))
        print('Time to compute train accuracy: ', (datetime.now()-t0))

        t0 = datetime.now()
        print('Testing Accuracy: ', knn.score(X_test, y_test))
        print('Time to compute test accuracy: ', (datetime.now()-t0))