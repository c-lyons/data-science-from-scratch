import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from datetime import datetime

## similar to naive bayes - but non-naive, i.e. independance is not assumed 


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

class NaiveBayes(object):

    def fit(self, X, y, smoothing=10e-3):
        self.gaussians = dict()
        self.priors = dict()
        classes = set(y)
        N, D = X.shape
        for c in classes:
            X_c = X[y==c]
            self.gaussians[c] = dict(
                                    mean=X_c.mean(axis=0),
                                    cov=np.cov(X_c.T)+np.eye(D)*smoothing
                                    )
            self.priors[c] = float(len(y[y==c]))/len(y)  # prior prob == rate of c in y

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P==y)

    def predict(self, X):
        N, D = X.shape  # getting shape of input X
        K = len(self.gaussians)  # num of classes
        P = np.zeros((N, K))  # initialising array to store probabilities for each class K for N samples

        for c, gauss in self.gaussians.items():
            mean, cov = gauss['mean'], gauss['cov']  # extracting gauss, var from gaussian dict for class c
            P[:, c] = multivariate_normal.logpdf(X, mean=mean, cov=cov, allow_singular=True) + np.log(self.priors[c])  # using logs so additive probs and not mult
        return np.argmax(P, axis=1)  # return label c which is max for row N

if __name__=='__main__':
    X, y = get_data(limit=10000)
    n_train = int(len(y)*0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    model = NaiveBayes()
    t_0 = datetime.now()
    model.fit(X_train, y_train)
    print('Time to fit model: ', datetime.now()-t_0)
    t_0 = datetime.now()
    print('Training accuracy: ', model.score(X_train, y_train))
    print('Time for train compute: ', datetime.now() - t_0)
    t_0 = datetime.now()
    print('Testing accuracy: ', model.score(X_test, y_test))
    print('Time for test compute: ', datetime.now() - t_0)