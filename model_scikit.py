import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


def function_approx(n_layers, func):
    n = 2000
    x = np.random.uniform(-15, 15, size = n)
    y = func(x)
    X = np.reshape(x ,[n, 1]) 
    y = np.reshape(y ,[n ,])

    clf = MLPRegressor(alpha=0.001, hidden_layer_sizes = (n_layers,), max_iter = 100000, 
                     activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')
    a = clf.fit(X, y)


    x_ = np.linspace(-10, 10, 160) # define axis

    pred_x = np.reshape(x_, [160, 1]) # [160, ] -> [160, 1]
    pred_y = a.predict(pred_x) # predict network output given x_
    fig = plt.figure() 
    plt.plot(x_, func(x_), color = 'b') # plot original function
    plt.plot(pred_x, pred_y, 'o', color = 'red') # plot network output
    plt.show()

function_approx(10, lambda x: (x)**3)
function_approx(10, lambda x: 3+ 4*(x)**4)