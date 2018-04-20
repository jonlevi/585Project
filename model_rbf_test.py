import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from model_scratch_rbf import RBFNetwork
from scipy.spatial.distance import euclidean

def generate_rectangles(num_imgs):
    global img_size
    min_object_size = 1
    max_object_size = 18

    imgs = np.zeros((num_imgs, img_size, img_size))

    for i in range(num_imgs):
        w, h = np.random.randint(min_object_size, max_object_size, size=2)

        x = int(img_size/2)
        y = int(img_size/2)

        imgs[i, x-w:x+w, y-h:y+h] = 1.

    return imgs

def pick_unique_imgs(imgs):
    x = np.unique(imgs, return_index=True, axis=0)
    unique_imgs = x[0]
    return unique_imgs

global img_size

# generate images of rectangle
N=1000
img_size = 50
imgs = pick_unique_imgs(generate_rectangles(N))

# Shuffle order of images
indx = [i for i in range(len(imgs))]
random.shuffle(indx)
imgs = imgs[indx]
print (imgs.shape)

# parse into train and test samples
train_size = int(len(imgs)*0.75)
test_size = len(imgs) - train_size
train_set = np.array([imgs[i].flatten() for i in range(train_size)])
test_set = np.array([imgs[i].flatten() for i in range(train_size, len(imgs))])

# set the first image to standard view
standard_view = train_set[0]

# Create network
# network takes target functions for each output unit
n_centers = 10
target_funcs = [(lambda i: standard_view[i]) for x in range(len(standard_view))]
n = RBFNetwork(n_in = img_size*img_size, 
               n_hidden = n_centers, 
               n_out = len(standard_view), 
               lr = 1, 
               h_funcs = target_funcs,
               beta = 0.005)
n.centers = train_set[:n_centers]

# train network
v = n.train(train_set, n_iter=1)

# test network
predicted_imgs = [n.activate(test_set[i]).reshape((img_size, img_size)) for i in range(test_size)]

# create shuffled test imges
shuffled_set = []
for img in test_set:
    arr = img.copy()
    subarr = arr[1200:1400].copy()
    np.random.shuffle(subarr)
    arr[1200:1400] = subarr
    shuffled_set.append(arr)

# test network on shuffled test imgs
predicted_shuffled_imgs = [n.activate(shuffled_set[i]).reshape((img_size, img_size)) for i in range(test_size)]

# plot results
vmin = np.min((np.min(predicted_imgs), np.min(predicted_shuffled_imgs)))
vmax = np.max((np.max(predicted_imgs), np.max(predicted_shuffled_imgs)))

for i in range(5):

    plt.figure()
    # plot output for the ith test image
    plt.subplot(2,3,1)
    plt.imshow(standard_view.reshape((img_size,img_size)).T, cmap='Greys', origin='lower')
    plt.subplot(2,3,2)
    plt.imshow(test_set[i].reshape((img_size,img_size)).T, cmap='Greys', origin='lower')
    plt.subplot(2,3,3)
    plt.imshow(predicted_imgs[i].T, vmin=vmin, vmax=vmax, cmap='Greys', origin='lower')

    # plot output for the ith shuffled test image
    plt.subplot(2,3,4)
    plt.imshow(standard_view.reshape((img_size,img_size)).T, cmap='Greys', origin='lower')
    plt.subplot(2,3,5)
    plt.imshow(shuffled_set[i].reshape((img_size,img_size)).T, cmap='Greys', origin='lower')
    plt.subplot(2,3,6)
    plt.imshow(predicted_shuffled_imgs[i].T, vmin=vmin, vmax=vmax, cmap='Greys', origin='lower')

    plt.gcf().set_size_inches(10, 10)
    plt.tight_layout()

plt.show()