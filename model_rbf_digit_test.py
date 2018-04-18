import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from model_scratch_rbf import RBFNetwork
from sklearn.datasets import load_digits


# Load handwritten digit dataset
all_images, digits = load_digits(n_class=10, return_X_y=True)
zeros = all_images[digits==0]
twos = all_images[digits==2]
threes = all_images[digits==3]
fours = all_images[digits==4]
imgs = fours

# Shuffle order of images
indx = [i for i in range(len(imgs))]
random.shuffle(indx)
imgs = imgs[indx]

# Create standard view based on an average drawing of the number
#standard_view = np.mean(imgs, axis=0)
#standard_view = np.mean(imgs > 0, axis=0)
standard_view = np.mean(imgs > 0, axis=0) > .9

# Set parameters
n_in = imgs.shape[1]
n_centers = 100
n_out = imgs.shape[1]
learning_rate = 1

# Run model
standard_view = np.mean(imgs[:n_centers], axis=0)
n = RBFNetwork(n_in, n_centers, n_out, learning_rate, [(lambda i: standard_view[i]) for x in range(len(standard_view))])
n.centers = imgs[:n_centers]
v = n.train(imgs, n_iter=3)

input_imgs = []
predicted_imgs = []
standard_view = standard_view.reshape((8, 8))
for i in range(5):
    img = imgs[-i]
    input_imgs.append(img.reshape((8, 8)))
    predict1 = n.v(img).sum(axis=1).reshape((8, 8))
    predicted_imgs.append(predict1)

    img = twos[-i]
    input_imgs.append(img.reshape((8, 8)))
    predict2 = n.v(img).sum(axis=1).reshape((8, 8))
    predicted_imgs.append(predict2)

vmin = np.min(predicted_imgs)
vmax = np.max(predicted_imgs)
vmin=None
vmax=None
for i in range(0, 10, 2):

    plt.subplot(10, 3, i*3+1)
    plt.imshow(standard_view, cmap='Greys', interpolation='none', extent=[0, 8, 0, 8])
    plt.subplot(10, 3, i*3+2)
    plt.imshow(input_imgs[i], cmap='Greys', interpolation='none', extent=[0, 8, 0, 8])
    plt.subplot(10, 3, i*3+3)

    predict1 = predicted_imgs[i]
    plt.imshow(predict1, cmap='Greys', vmin=vmin, vmax=vmax, interpolation='none', extent=[0, 8, 0, 8])
    # print ('original pic', euclidean(standard_view, predict1.flatten()))

    j = i + 1
    plt.subplot(10, 3, j * 3 + 1)
    plt.imshow(standard_view, cmap='Greys', interpolation='none', extent=[0, 8, 0, 8])
    plt.subplot(10, 3, j * 3 + 2)
    arr = input_imgs[j]
    plt.imshow(arr, cmap='Greys', interpolation='none', extent=[0, 8, 0, 8])
    plt.subplot(10, 3, j * 3 + 3)

    predict2 = predicted_imgs[j]
    plt.imshow(predict2, cmap='Greys', vmin=vmin, vmax=vmax, interpolation='none', extent=[0, 8, 0, 8])
    # print('random-ish: ', euclidean(standard_view, predict2.flatten()))

plt.gcf().set_size_inches(5, 10)
plt.tight_layout()
plt.show()
pass

