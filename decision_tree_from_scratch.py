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

class TreeNode:

    def __init__(self, depth, max_depth):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X,y):
        if len(y)==1 or len(set(y))==1:  # base case of len(y)==1  or only 1 class in y
            self.col = None
            self.split = None
            self.left = None
            self.right = None
        else:
            N, D = X.shape
            cols = range(D)
            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, y, col)
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split
            if max_ig==0:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(y.mean())
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    # if max depth reached set split nodes to None
                    self.left = None
                    self.right= None
                    # getting majority class on split
                    self.prediction = [np.round(y[X[:, best_col] < self.split].mean()),
                                       np.round(y[X[:, best_col] >= self.split].mean())]
                else: # else not base case -> recursion
                    left_idx = (X[:, best_col] < best_split)
                    X_left = X[left_idx]
                    y_left = y[left_idx]
                    self.left = TreeNode(self.depth+1, self.max_depth)
                    self.left.fit(X_left, y_left)

                    right_idx = (X[:, best_col] >= best_split)
                    X_right = X[right_idx]
                    y_right = y[right_idx]
                    self.right = TreeNode(self.depth+1, self.max_depth)
                    self.right.fit(X_right, y_right)

    def find_split(self, X, y, col):
        x_vals = X[:, col]
        sort_idx = np.argsort(x_vals)
        x_vals = x_vals[sort_idx]
        y_vals = y[sort_idx]
        # get boundaries of where sorted labels change value
        #  i.e. get index where a != b for array(a, a, a, a, b, b, b, b) == index 3
        boundaries = np.nonzero(y_vals[:-1] != y_vals[1:])[0]
        best_split = None
        max_ig = 0
        for i in boundaries:
            split = (x_vals[i] + x_vals[i+1])/2  # setting split to between boundary values
            ig = self.information_gain(x_vals, y_vals, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, X, y, split):
        y_0 = y[X < split]
        y_1 = y[X >= split]
        N = len(y)
        y_0_len = len(y_0)
        if y_0_len == 0 or y_0_len == N:
            return 0 # base case - no IG for all same class/ no class below split
        p_0 = float(len(y_0))/N  # P0 = proportion of y_0
        p_1 = 1 - p_0  # P1 = 1 - P0
        return entropy(y) - p_0*entropy(y_0) - p_1*(entropy(y_1))

    def predict_one(self, X):
        if self.col is not None and self.split is not None:
            # i.e. if we have a split
            feature = X[self.col]
            if feature < self.split:
                # if less than split we go to left side
                if self.left:
                    # if self.left exists then recursive call to predict one
                    p = self.left.predict_one(X)
                else:
                    # else return leaf node prediction
                    p = self.prediction[0]
            else:
                if self.right:
                    # if right child exists recursive call to predict one
                    p = self.right.predict_one(X)
                else:
                    # else return prediction for leaf node
                    p = self.prediction[1]
        else:
            p = self.prediction  # other base case where only 1 prediction for leaf node
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = TreeNode(depth=None, max_depth=self.max_depth)
        self.root.fit(X, y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P==y)

def entropy(y):
    N = len(y)
    s1 = (y==1).sum()
    if 0 == s1 or N == s1:
        return 0  # base case for all classes being same
    p1 = float(s1)/N
    p0 = 1-p1
    return -p0*np.log2(p0) - p1*np.log2(p1)


if __name__=='__main__':
    X, y = get_data(10000)
    # limiting data to binary case of y = 0 r y =1 only and ignoring other y values
    idx = np.logical_or(y==0, y==1)
    X = X[idx]
    y = y[idx]
    N_train = int(len(y)*0.8)
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]

    model = DecisionTree()
    t_0 = datetime.now()
    model.fit(X_train, y_train)
    print('Time to fit model: ', datetime.now()-t_0)
    t_0 = datetime.now()
    print('Training accuracy: ', model.score(X_train, y_train))
    print('Time for train compute: ', datetime.now() - t_0)
    t_0 = datetime.now()
    print('Testing accuracy: ', model.score(X_test, y_test))
    print('Time for test compute: ', datetime.now() - t_0)