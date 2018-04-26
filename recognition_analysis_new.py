import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from model_scratch_rbf import RBFNetwork
from sklearn.datasets import load_digits
from scipy.spatial.distance import euclidean

def test_wrapper(NUMBER_TO_TRAIN, n_centers, train_size):

  # Set parameters
  n_in = 64
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
  train_set = imgs[:train_size]
  test_set = imgs[train_size:]
  v = n.train(train_set, n_iter=n_training_iter)

  # Test on all images that were not used for training
  predicted_imgs = [None for i in range(10)]
  activation = [None for i in range(10)]
  distances = [None for i in range(10)]
  for i in range(10):
      if i == NUMBER_TO_TRAIN:
          predicted_imgs[i] = np.array([n.activate(img) for img in test_set])
      else:
          predicted_imgs[i] = np.array([n.activate(img) for img in all_images[i]])
      activation[i] = np.max(predicted_imgs[i], axis=1)
      distances[i] = [euclidean(img, standard_view) for img in predicted_imgs[i]]

  # determine threshold on training activations
  train_activation = [n.activate(img) for img in train_set]
  threshold = np.min([np.max(x) for x in train_activation])
  response = [x >= threshold for x in activation]

  # Calculate the percent of the time the model responds "yes" to each digit type
  percent_yes = [x.mean() for x in response]
  # print(percent_yes)

  # calculate hit and false alarm rate
  this_number_test_response = response[NUMBER_TO_TRAIN]
  hit_rate = np.mean(this_number_test_response)
  other_numbers_test_response = []
  for i in range(10):
    if i!=NUMBER_TO_TRAIN:
      other_numbers_test_response = other_numbers_test_response + response[i].tolist()
  fa_rate = np.mean(other_numbers_test_response)
  sensitivity = hit_rate - fa_rate

  # print ('hit rate=%0.3f   fa rate=%0.3f' % (hit_rate, fa_rate))
  # print ('sensitivity=%0.3f'% sensitivity)

  return sensitivity

NUMBER_TO_TRAIN = 0
n_loops = 100

# test performance with varying centers
train_size = 50
scores = []
for n_centers in range(2,21):
  print ('n_center: ',n_centers)
  sensitivities = [test_wrapper(NUMBER_TO_TRAIN, n_centers, train_size) for i in range(n_loops)]
  scores.append(sensitivities)
  print ('mean sensitivity: ', np.mean(sensitivities))
scores = np.array(scores)

print (scores.shape)
print (np.mean(scores).mean(1))

# # test performance with varying centers
# n_centers = 15
# scores = []
# for train_size in range(5,105,5):
#   sensitivities = [test_wrapper(NUMBER_TO_TRAIN, n_centers, train_size) for i in range(n_loops)]
#   scores.append(sensitivities)
# scores = np.array(scores)