import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from model_scratch_rbf import RBFNetwork
from scipy.spatial.distance import euclidean



def generate_imgs(num_imgs):
    global img_size
    min_object_size = 1
    max_object_size = 18

    bboxes = np.zeros((num_imgs, 4))
    imgs = np.zeros((num_imgs, img_size, img_size))

    for i in range(num_imgs):
        w, h = np.random.randint(min_object_size, max_object_size, size=2)

        x = int(img_size/2)
        y = int(img_size/2)

        imgs[i, x-w:x+w, y-h:y+h] = 1.
        bboxes[i] = [x-w, y-h, w*2, h*2]

    return [imgs, bboxes]

def pick_unique_imgs(imgs, bboxes):
    x = np.unique(imgs, return_index=True, axis=0)
    unique_imgs = x[0]
    unique_bboxes = bboxes[x[1], :]

    return unique_imgs, unique_bboxes

global bboxes, img_size

N=1000
img_size = 50
imgs, bboxes = pick_unique_imgs(*generate_imgs(N))

# Shuffle order of images
indx = [i for i in range(len(imgs))]
random.shuffle(indx)
imgs = imgs[indx]
bboxes = bboxes[indx]
print (imgs.shape)

# i = 0
# plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
# plt.gca().add_patch(matplotlib.patches.Rectangle((bboxes[i][0],bboxes[i][1]),bboxes[i][2],bboxes[i][3], ec='r', fc='none'))
# plt.show()

def h(s):
    global bboxes, img_size
    s = s.reshape(img_size, img_size)
    for i in range(len(imgs)):
        if np.all(imgs[i] == s):
            break
    return bboxes[i]

def hx(s):
    return h(s)[0]

def hy(s):
    return h(s)[1]

def hw(s):
    return h(s)[2]

def hh(s):
    return h(s)[3]

train_size = int(len(imgs)*0.75)
test_size = len(imgs) - train_size
train_set = np.array([imgs[i].flatten() for i in range(train_size)])
test_set = np.array([imgs[i].flatten() for i in range(train_size, len(imgs))])
standard_view = train_set[0]

#n = RBFNetwork(img_size*img_size, 15, 4, 5, [hx, hy, hw, hh])
n_centers = 10
n = RBFNetwork(img_size*img_size, n_centers, len(standard_view), 1, [(lambda i: standard_view[i]) for x in range(len(standard_view))])
n.centers = train_set[:n_centers]
v = n.train(train_set, n_iter=1)

predicted_imgs = []
input_imgs = []

for i in [0,1,2,3,4]:

    input_imgs.append(test_set[-i])
    predict1 = n.v(test_set[-i]).sum(axis=1).reshape((img_size, img_size))
    predicted_imgs.append(predict1)

    arr = test_set[-i].copy()
    subarr = arr[1200:1400].copy()
    np.random.shuffle(subarr)
    arr[1200:1400] = subarr
    input_imgs.append(arr)
    predict2 = n.v(arr).sum(axis=1).reshape((img_size, img_size))
    predicted_imgs.append(predict2)


print (np.array(input_imgs).shape)

vmin = np.min(predicted_imgs)
vmax = np.max(predicted_imgs)

for i in [0,2,4,6,8]:

    plt.subplot(10,3,i*3+1)
    plt.imshow(standard_view.reshape((img_size,img_size)).T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    plt.subplot(10,3,i*3+2)
    plt.imshow(input_imgs[i].reshape((img_size,img_size)).T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    plt.subplot(10,3,i*3+3)

    predict1 = predicted_imgs[i]
    plt.imshow(predict1.T, cmap='Greys', vmin=vmin, vmax=vmax, interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    # print ('original pic', euclidean(standard_view, predict1.flatten()))

    j = i + 1
    plt.subplot(10, 3, j * 3 + 1)
    plt.imshow(standard_view.reshape((img_size, img_size)).T, cmap='Greys', interpolation='none', origin='lower',
               extent=[0, img_size, 0, img_size])
    plt.subplot(10, 3, j * 3 + 2)
    arr = input_imgs[j]
    plt.imshow(arr.reshape((img_size, img_size)).T, cmap='Greys', interpolation='none', origin='lower',
               extent=[0, img_size, 0, img_size])
    plt.subplot(10, 3, j * 3 + 3)

    predict2 = predicted_imgs[j]
    plt.imshow(predict2.T, cmap='Greys', vmin=vmin, vmax=vmax, interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    # print('random-ish: ', euclidean(standard_view, predict2.flatten()))

plt.gcf().set_size_inches(5,10)
plt.tight_layout()
plt.show()


pass