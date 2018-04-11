import random
import numpy as np
import matplotlib.pyplot as plt
from cordic import cossin_cordic
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit


class Network:

    def gauss(self, s, node):
        return np.exp(-self.betas[node]*(s-self.centers[node])**2)
    def rect(self, s, node):
        center = self.centers[node]
        if abs(center-s) < 1.5:
            return 1
        else:
            return 0

    def v(self, s):
        return np.dot(self.w, np.array([self.f(s, i) for i in range(self.n_in)]))

    def update_weights(self, s):
        target = np.array([self.h[i](s) for i in range(self.n_out)])

        diff = target - self.v(s)
        self.w += self.learning_rate * np.dot(np.atleast_2d(diff).T, 
                np.array([[self.f(s, i) for i in range(self.n_in)]]))
        # self.betas += self.learning_rate * 0.0001 * np.dot(np.atleast_2d(diff).T, np.array([[self.f(s, i) for i in range(self.n_in)]])).mean(axis=0)
        # self.centers += self.learning_rate * np.dot(np.atleast_2d(diff).T, np.array([[self.f(s, i) for i in range(self.n_in)]])).mean(axis=0)
    def train(self, stim):
        sorted_stim = sorted(stim)
        x = sorted_stim[::100]
        v1 = np.zeros((100, 100))
        v2 = np.zeros((100, 100))
        for i, s in enumerate(stim):

            self.update_weights(s)
            if i % 100 == 0:
                v1[int(i/100),:] = [self.v(s)[0] for s in x]
                v2[int(i/100),:] = [self.v(s)[1] for s in x]
        v1 = np.array(v1)
        v2 = np.array(v2)

        return [v1, v2]

    def __init__(self, n_in, n_out, lr, h_funcs, gaussian = True):



        self.n_in = n_in
        self.n_out = n_out
        self.learning_rate = lr
        self.h = h_funcs
        self.w = np.random.rand(n_in * n_out).reshape((n_out, n_in))
        # self.betas = np.random.rand(n_in)
        self.betas = [0.5] * n_in
        # self.centers = 10 * np.random.rand(n_in)
        self.centers = np.linspace(10, -10, num=n_in)
        if (gaussian):
            self.f = self.gauss
        else:
    


            self.f = self.rect

def exp_func(x, a, b, c):
    return a*np.exp(-b*x) + c


def speed_vs_accuracy():
    mses_cos = []
    mses_sin = []
    for i in range(1,25):
        N = Network(i, 2, .005, [np.cos, np.sin])
        stim = [random.uniform(-2*np.pi, 2*np.pi) for i in range(10000)]
        sorted_stim = sorted(stim)
        x = sorted_stim[::100]
        results = N.train(stim)
        v1 = np.array(results[0])
        # print(v1)
        v2 = np.array(results[1])

        mses_cos.append(mean_squared_error(np.cos(x), v1[70,:]))
        mses_sin.append(mean_squared_error(np.sin(x), v2[70,:]))

    plt.figure(figsize=(20,10))
    plt.title('Speed Accuracy Tradeoff in Size of Network')
    plt.subplot(121)
    plt.plot(range(1,25), mses_cos)
    popt, pcov = curve_fit(exp_func, range(1,25), mses_cos)
    plt.plot(range(1,25), exp_func(range(1,25), *popt),'--r')
    plt.xlabel('Number of Neurons in Hidden Layer')
    plt.ylabel('MSE of Function Approximation')
    plt.title('cos(x) Approximation')
    

    plt.subplot(122)
    plt.plot(range(1,25), mses_sin)
    popt, pcov = curve_fit(exp_func, range(1,25), mses_sin)
    plt.plot(range(1,25), exp_func(range(1,25), *popt),'--r')
    plt.xlabel('Number of Neurons in Hidden Layer')
    plt.title('sin(x) Approximation')
    plt.savefig('tradeoff.png')


def guass_network():

    N = Network(11, 2, .005, [np.cos, np.sin], True)
    stim = [random.uniform(-2*np.pi, 2*np.pi) for i in range(10000)]
    sorted_stim = sorted(stim)
    x = sorted_stim[::100]
    results = N.train(stim)
    v1 = np.array(results[0])
    # print(v1)
    v2 = np.array(results[1])
    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(x, np.cos(x), 'k', label='Target Function')
    plt.plot(x, v1[10,:], '--', lw=0.75, label='1000th')
    plt.plot(x, v1[30,:], '--', lw=0.75, label='3000th')
    plt.plot(x, v1[50,:], '--', lw=0.75, label='5000th')
    plt.plot(x, v1[70,:], '--', lw=0.75, label='7000th')
    plt.title('Approximating cos(x)')
    plt.legend()

    plt.subplot(122)
    plt.plot(x, np.sin(x), 'k', label='Target Function')
    plt.plot(x, v2[10,:], '--', lw=0.75, label='1000th')
    plt.plot(x, v2[30,:], '--', lw=0.75, label='3000th')
    plt.plot(x, v2[50,:], '--', lw=0.75, label='5000th')
    plt.plot(x, v2[70,:], '--', lw=0.75, label='7000th')
    plt.title('Approximating sin(x)')
    plt.legend()
    plt.savefig('gaussian.png')



def delta_network():
    N = Network(11, 2, .005, [np.cos, np.sin], False)
    stim = [random.uniform(-2*np.pi, 2*np.pi) for i in range(10000)]
    sorted_stim = sorted(stim)
    x = sorted_stim[::100]
    results = N.train(stim)
    v1 = np.array(results[0])
    # print(v1)
    v2 = np.array(results[1])

    plt.figure(figsize=(30,10))
    plt.subplot(221)
    plt.plot(x, np.cos(x), 'k', label='Target Function')
    plt.plot(x, v1[10,:], '--', lw=0.75, label='1000th')
    plt.plot(x, v1[30,:], '--', lw=0.75, label='3000th')
    plt.plot(x, v1[50,:], '--', lw=0.75, label='5000th')
    plt.plot(x, v1[70,:], '--', lw=0.75, label='7000th')
    plt.title('Approximating cos(x) with Delta Function Filter')
    plt.legend()

    plt.subplot(222)
    plt.plot(x, np.sin(x), 'k', label='Target Function')
    plt.plot(x, v2[10,:], '--', lw=0.75, label='1000th')
    plt.plot(x, v2[30,:], '--', lw=0.75, label='3000th')
    plt.plot(x, v2[50,:], '--', lw=0.75, label='5000th')
    plt.plot(x, v2[70,:], '--', lw=0.75, label='7000th')
    plt.title('Approximating sin(x)with Delta Function Filter')
    plt.legend()

    plt.subplot(223)
    cord = np.array([cossin_cordic(a,3) for a in x])
    plt.plot(x, cord[:,0], label = 'cos')
    plt.title('cos(x) via 3 Iterations on CORDIC Approximation')
    plt.subplot(224)
    plt.title('sin(x) via 3 Iterations on CORDIC Approximation')
    plt.plot(x, cord[:,1], label = 'sin')


    plt.savefig('delta.png')

def main():
    #training network with 11 nodes to approximate sin and cos functions
    #1: Test Different Network Sizes using Gaussian Tuning Curves to see speed/accuracy tradeoff
    speed_vs_accuracy()

    #train with 11 nodes using Gaussian Tuning Curves
    guass_network()

    #train with 11 nodes using Delta Function Tuning Curves
    delta_network()


if __name__ == "__main__": main()
