import random
import numpy as np
import matplotlib.pyplot as plt


class Network:

    def __init__(self, n_in, n_out, lr, h_funcs):

        self.n_in = n_in
        self.n_out = n_out
        self.learning_rate = lr
        self.h = h_funcs
        self.w = np.random.rand(n_in * n_out).reshape((n_out, n_in))
        # self.betas = np.random.rand(n_in)
        self.betas = [0.5] * n_in
        # self.centers = 10 * np.random.rand(n_in)
        self.centers = np.linspace(10, -10, num=n_in)

    def f(self, s, node):
        return np.exp(-self.betas[node]*(s-self.centers[node])**2)

    def v(self, s):
        return np.dot(self.w, np.array([self.f(s, i) for i in range(self.n_in)]))

    def update_weights(self, s):
        target = np.array([self.h[i](s) for i in range(self.n_out)])
        diff = target - self.v(s)
        self.w += self.learning_rate * np.dot(np.atleast_2d(diff).T, np.array([[self.f(s, i) for i in range(self.n_in)]]))
        # self.betas += self.learning_rate * 0.0001 * np.dot(np.atleast_2d(diff).T, np.array([[self.f(s, i) for i in range(self.n_in)]])).mean(axis=0)
        # self.centers += self.learning_rate * np.dot(np.atleast_2d(diff).T, np.array([[self.f(s, i) for i in range(self.n_in)]])).mean(axis=0)


N = Network(11, 2, .005, [np.cos, np.sin])
stim = [random.uniform(-2*np.pi, 2*np.pi) for i in range(10000)]
sorted_stim = sorted(stim)
x = sorted_stim[::100]
print(N.w)

v1 = np.zeros((100, 100))
v2 = np.zeros((100, 100))
for i, s in enumerate(stim):
    N.update_weights(s)

    if i % 100 == 0:
        v1[int(i/100),:] = [N.v(s)[0] for s in x]
        v2[int(i/100),:] = [N.v(s)[1] for s in x]
    #print(N.w)

v1 = np.array(v1)
v2 = np.array(v2)

plt.subplot(121)
plt.plot(x, np.cos(x), 'k')
plt.plot(x, v1[10,:], '--', lw=0.75, label='1000th')
plt.plot(x, v1[30,:], '--', lw=0.75, label='3000th')
plt.plot(x, v1[50,:], '--', lw=0.75, label='5000th')
plt.plot(x, v1[70,:], '--', lw=0.75, label='7000th')

plt.subplot(122)
plt.plot(x, np.sin(x), 'k')
plt.plot(x, v2[10,:], '--', lw=0.75, label='1000th')
plt.plot(x, v2[30,:], '--', lw=0.75, label='3000th')
plt.plot(x, v2[50,:], '--', lw=0.75, label='5000th')
plt.plot(x, v2[70,:], '--', lw=0.75, label='7000th')

plt.legend()
