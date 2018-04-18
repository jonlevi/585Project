import random
import numpy as np
import matplotlib.pyplot as plt
from cordic import cossin_cordic
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean


class RBFNetwork:

    def gauss(self, s, node):
        return np.exp(-self.betas[node]*euclidean(s, self.centers[node])**2)

    def v(self, s):
        # return np.dot(self.w, np.array([self.f(s, i) for i in range(self.n_hidden)]))
        # return np.dot(self.w, np.array([self.f(s, i) * self.centers[i] for i in range(self.n_hidden)]))
        return self.w * np.array([self.f(s, i) for i in range(self.n_hidden)]).T  # n_out x n_hidden

    def add_center(self, s):
        nan_locations = np.where(np.isnan(self.centers))[0]
        if len(nan_locations) > 0:
            # Find the first row (i.e. center) with a NaN and set the center equal to the current stimulus
            self.centers[nan_locations[0], :] = s

    def update_weights(self, s):
        target = np.array([self.h[i](i) for i in range(self.n_out)])  # n_out
        v = self.v(s)
        #diff = np.array([target - v[:, i] for i in range(self.n_hidden)]).T
        #self.w += self.learning_rate * np.dot(diff, np.array([[self.f(s, i) for i in range(self.n_hidden)]]).T)
        diff = target - v.sum(axis=1)
        self.w += self.learning_rate * np.dot(np.atleast_2d(diff).T, np.array([[self.f(s, i) for i in range(self.n_hidden)]]))

    def train(self, stim, n_iter=1):
        v = np.empty((len(stim)*n_iter, self.n_out))
        for n in range(n_iter):
            for i, s in enumerate(stim):
                if np.any(np.isnan(self.centers)):
                    self.add_center(s)
                else:
                    self.update_weights(s)
                    v[i, :] = self.v(s).sum(axis=1)
        return v

    def __init__(self, n_in, n_hidden, n_out, lr, h_funcs):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.learning_rate = lr
        self.h = h_funcs
        self.w = np.random.rand(n_hidden * n_out).reshape((n_out, n_hidden))
        self.betas = [.5] * n_hidden
        self.centers = np.empty((n_hidden, n_in))
        self.centers.fill(np.nan)
        self.f = self.gauss

def exp_func(x, a, b, c):
    return a*np.exp(-b*x) + c

def main():
    pass


if __name__ == "__main__": main()
