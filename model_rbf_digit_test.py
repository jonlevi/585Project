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
fives = all_images[digits==5]
sixes = all_images[digits==6]
sevens = all_images[digits==7]
eights = all_images[digits==8]
nines = all_images[digits==9]

# select which number to train the network on
imgs = sixes

# Shuffle order of images
indx = [i for i in range(len(imgs))]
random.shuffle(indx)
imgs = imgs[indx]

# parse into train and test samples
train_size = int(len(imgs)*0.75)
test_size = len(imgs) - train_size
train_set = np.array([imgs[i].flatten() for i in range(train_size)])
test_set = np.array([imgs[i].flatten() for i in range(train_size, len(imgs))])

# Set parameters
n_in = imgs.shape[1]
n_centers = 100
n_out = imgs.shape[1]
learning_rate = 1
beta = 0.005

# Run model
standard_view = np.mean(imgs[:n_centers], axis=0)
target_funcs = [(lambda i: standard_view[i]) for x in range(len(standard_view))]
n = RBFNetwork(n_in = n_in, 
               n_hidden = n_centers, 
               n_out = n_out, 
               lr = learning_rate,
               h_funcs = target_funcs,
               beta = 0.005)
n.centers = imgs[:n_centers]
v = n.train(imgs, n_iter=3)

img_size = 8

# test on unseen images of the selected number
predicted_imgs = [n.activate(test_set[i]).reshape((img_size, img_size)) for i in range(test_size)]

# test on images of other numbers
test_other = fours
other_imgs = [n.activate(test_other[i]).reshape((img_size, img_size)) for i in range(test_size)]

# plot results
vmin = np.min((np.min(predicted_imgs), np.min(other_imgs)))
vmax = np.max((np.max(predicted_imgs), np.max(other_imgs)))

for i in range(5):

    plt.figure()
    # plot output for the ith test image
    plt.subplot(2,3,1)
    plt.imshow(standard_view.reshape((img_size,img_size)), cmap='Greys')
    plt.subplot(2,3,2)
    plt.imshow(test_set[i].reshape((img_size,img_size)), cmap='Greys')
    plt.subplot(2,3,3)
    plt.imshow(predicted_imgs[i], vmin=vmin, vmax=vmax, cmap='Greys')

    # plot output for the ith shuffled test image
    plt.subplot(2,3,4)
    plt.imshow(standard_view.reshape((img_size,img_size)), cmap='Greys')
    plt.subplot(2,3,5)
    plt.imshow(test_other[i].reshape((img_size,img_size)), cmap='Greys')
    plt.subplot(2,3,6)
    plt.imshow(other_imgs[i], vmin=vmin, vmax=vmax, cmap='Greys')

    plt.gcf().set_size_inches(10, 10)
    plt.tight_layout()

plt.show()
