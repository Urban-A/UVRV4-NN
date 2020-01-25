import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import matplotlib.image as mpimg
from scipy.ndimage import rotate
from scipy.misc import face
#import cv2

data = keras.datasets.fashion_mnist

#(train_images, train_labels), (test_images, test_labels) = data.load_data();

class_names_t = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names = ['soda', 'liha']

makeNew=True
nOfImages=200

#train_images = np.array([np.array(np.mean(mpimg.imread("train/train_"+str(number)+".png")[:,:,:3][:, :, :3],2)) for number in range(nOfImages)])

train_images = np.array([np.array(mpimg.imread("train/train_"+str(number)+".png")[:,:,:3]) for number in range(nOfImages)])
test_images = np.array([np.array(mpimg.imread("test/test_"+str(number)+".png")[:,:,:3]) for number in range(nOfImages)])

test_images_s = test_images[1::2]
test_images_l = test_images[::2]

#train_images=train_images_alpha[:,:,:,:3]
#test_images=test_images_alpha[:,:,:,:3]

#train_images=np.delete(train_images, slice(None, None, None),slice(None, None, None) ,4)

#print(train_images)

train_labels=np.zeros(nOfImages)
test_labels=np.zeros(nOfImages)

test_labels_s = test_labels[1::2]
test_labels_l = test_labels[::2]

train_labels[0::2]=1
test_labels[0::2]=1

#"""
for i in range(nOfImages):
    for j in range(40):
        if makeNew:
            temp_image = rotate(train_images[i], random.randint(0, 359), reshape=False, order=1)
            temp_image = np.roll(temp_image, random.randint(-1,1), random.randint(0,1))

            mpimg.imsave("augmented/aug_"+str(i)+"_"+str(j)+".png", temp_image)
            #imageio.imwrite("augmented/aug_"+str(i)+"_"+str(j)+".png", temp_image)

            train_labels = np.append(train_labels, train_labels[i])
            temp_image=[temp_image]
            train_images = np.append(train_images, temp_image, axis=0)

            print("i:",i ,", j:", j)
        else:
            temp_image=np.array(mpimg.imread("augmented/aug_"+str(i)+"_"+str(j)+".png"))
            train_labels = np.append(train_labels, train_labels[i])
            temp_image=[temp_image]
            train_images = np.append(train_images, temp_image, axis=0)


#"""

#train_images = train_images/255.0
#test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Convolution2D(64, 3, 3, activation='relu', input_shape=(64,64,3)),
    #keras.layers.Flatten(input_shape=(64,64,3)),
    keras.layers.Convolution2D(32, 3, 3, activation='relu'),
    keras.layers.Convolution2D(16, 3, 3, activation='relu'),
    keras.layers.Flatten(),
    #keras.layers.Dense(64, activation="relu"),
    #keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50)

test_loss, test_acc = model.evaluate(test_images_s, test_labels_s)
print("Tested Acc soda: ", test_acc)

test_loss, test_acc = model.evaluate(test_images_l, test_labels_l)
print("Tested Acc liha: ", test_acc)

"""
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)


test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Tested Acc: ", test_acc)


nombar=0
im = mpimg.imread("train/train_"+str(nombar)+".png")
plt.imshow(im)
plt.show()

#array0 = np.empty( , dtype=int)

#print(train_images)

print(test_labels)


nombar = 199
plt.imshow(train_images[nombar])
plt.show()

"""
