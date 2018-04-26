import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from model_scratch_rbf import RBFNetwork
from sklearn.datasets import load_digits

# Set parameters
NUMBER_TO_TRAIN = 0
n_in = 64
n_centers = 15
n_out = 64
learning_rate = 1
beta = .5
n_training_iter = 1

# Load handwritten digit dataset, normalize values (from 0-16 to 0-1), and shuffle order
all_images, digits = load_digits(n_class=10, return_X_y=True)
all_images = all_images / 16
indx = [i for i in range(len(all_images))]
random.shuffle(indx)
all_images = all_images[indx]
digits = digits[indx]

# Group images by their digits
zeros = all_images[digits==0]
ones = all_images[digits==1]
twos = all_images[digits==2]
threes = all_images[digits==3]
fours = all_images[digits==4]
fives = all_images[digits==5]
sixes = all_images[digits==6]
sevens = all_images[digits==7]
eights = all_images[digits==8]
nines = all_images[digits==9]
all_images = [zeros, ones, twos, threes, fours, fives, sixes, sevens, eights, nines]

# Initialize model
standard_view = np.mean(all_images[NUMBER_TO_TRAIN], axis=0)
target_funcs = [(lambda i: standard_view[i]) for x in range(len(standard_view))]
n = RBFNetwork(n_in=n_in,
               n_hidden=n_centers,
               n_out=n_out,
               lr=learning_rate,
               h_funcs=target_funcs,
               beta=beta)

# Split data into training and test sets and run training
imgs = all_images[NUMBER_TO_TRAIN]
train_size = int(len(imgs)*0.75)
train_set = imgs[:train_size]
test_set = imgs[train_size:]
v = n.train(train_set, n_iter=n_training_iter)

# Test on all images that were not used for training
predicted_imgs = [None for i in range(10)]
activation = [None for i in range(10)]
for i in range(10):
    if i == NUMBER_TO_TRAIN:
        predicted_imgs[i] = np.array([n.activate(img) for img in test_set])
    else:
        predicted_imgs[i] = np.array([n.activate(img) for img in all_images[i]])
    activation[i] = np.max(predicted_imgs[i], axis=1)

# Threshold will be set in a way that gives the hit rate defined below (provide hit rate as a percent, not a ratio)
HIT_RATE = 95
threshold = np.percentile(activation[NUMBER_TO_TRAIN], 100-HIT_RATE)
# Threshold activation levels to produce yes/no responses
response = [x >= threshold for x in activation]
# Calculate the percent of the time the model responds "yes" to each digit type
percent_yes = [x.mean() for x in response]
print(percent_yes)
