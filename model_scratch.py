import random
import numpy as np
import matplotlib.pyplot as plt


class Network:
    def __init__(self, n_nodes, lr, f_func, h_func):
        self.n_nodes = n_nodes
        self.learning_rate = lr
        self.h = h_func
        self.f = f_func
        self.w = np.random.rand(n_nodes)

    def v(self, s):
        return np.dot(self.w, np.array([self.f(s, i) for i in range(self.n_nodes)]))

    def update_weights(self, stimulus):
        self.w += self.learning_rate * (self.h(stimulus) - self.v(stimulus)) * \
                  np.array([self.f(stimulus, i) for i in range(self.n_nodes)])


def f(stim, node):
    param = -10 + 2 * node
    return np.exp(-0.5*(stim-param)**2)


N = Network(11, .005, f, np.cos)
stim = [random.uniform(-2 * np.pi, 2 * np.pi) for i in range(10000)]
sorted_stim = sorted(stim)
x = sorted_stim[::100]
print(N.w)

v = np.zeros((100, 100))
for i, s in enumerate(stim):
    N.update_weights(s)

    if i % 100 == 0:
        v[int(i / 100), :] = [N.v(s) for s in x]
        # print(N.w)

v = np.array(v)

plt.plot(x, np.cos(x), 'k')
plt.plot(x, v[10,:], '--', lw=0.75, label='1000th')
plt.plot(x, v[30,:], '--', lw=0.75, label='3000th')
plt.plot(x, v[50,:], '--', lw=0.75, label='5000th')
plt.plot(x, v[70,:], '--', lw=0.75, label='7000th')

plt.legend()
plt.show()

