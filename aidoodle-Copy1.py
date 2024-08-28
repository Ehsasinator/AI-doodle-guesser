def read_classes():
  f = open ("/Users/ehsanjaveddeveloper/Desktop/categories.txt", "r")
  out_classes = f.readlines()
  f.close()
  for i in range(0, len(out_classes)):
    out_classes[i] = out_classes[i].replace('\n', '').replace(' ', '_')
  return out_classes

## Call
out_classes = read_classes()
print (out_classes)

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers

## A function to load the data and split it into training and test sets.
def load_dataset():
  ## Initialize variables
  X = np.empty([0, 784])
  Y = np.empty([0, 1])
  images_per_class = 12000
  test_set_size = 10000
  
  ## Load the dataset
  for i in range(0, len(out_classes)):
    imgs = np.load(("/Users/ehsanjaveddeveloper/Desktop/"+out_classes[i] + ".npy"))   # Load images of a given doodle
    imgs = imgs[0 : images_per_class, :]        # Select the first 12000 images
    labels = np.full((images_per_class, 1), i)  # Create labels for the given doodle
    X = np.concatenate((X, imgs), axis = 0)     # Concatenate examples of each doodle
    Y = np.concatenate((Y, labels), axis = 0)   # Concatenate the labels
    del imgs          # Take extra care to make sure we don't run out of memory.

  ## Randomise the dataset
  np.random.seed(1)
  order = np.random.permutation(Y.shape[0])
  X = X[order, :]
  Y = Y[order, :]
  
  ## Split the data into training and test sets
  X_test = X[0 : test_set_size, :]
  Y_test = Y[0 : test_set_size, :]
  X_train = X[test_set_size : X.shape[0], :]
  Y_train = Y[test_set_size : Y.shape[0], :]
  
  return X_train, Y_train, X_test, Y_test


## Load the dataset
X_train, Y_train, X_test, Y_test = load_dataset()

## Sanity check the shape of out input.
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)

## Sanity check the shape of out input.
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)

## Reshape image vectors.
image_size = 28
X_train = X_train.reshape(X_train.shape[0], image_size, image_size, 1)
X_test = X_test.reshape(X_test.shape[0], image_size, image_size, 1)

## Pad the images to centre the content
X_train = np.pad(X_train, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
X_test = np.pad(X_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')

## Create one hot vectors for class labels.
Y_train = keras.utils.to_categorical(Y_train, len(out_classes))
Y_test = keras.utils.to_categorical(Y_test, len(out_classes))

## Normalize the dataset
X_train = X_train / 255
X_test = X_test / 255

## Sanity check the shape of data-sets.
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)


def create_model(input_shape):
  model = keras.Sequential()
  
  model.add(layers.Conv2D(6, (5, 5), input_shape = input_shape, activation = 'relu'))
  model.add(layers.BatchNormalization(axis = 3))
  model.add(layers.MaxPooling2D(pool_size = (2, 2)))
  
  model.add(layers.Conv2D(16, (5, 5), activation = 'relu'))
  model.add(layers.BatchNormalization(axis = 3))
  model.add(layers.MaxPooling2D(pool_size = (2, 2)))

  model.add(layers.Flatten())
  model.add(layers.Dense(120, activation = 'relu'))
  model.add(layers.Dense(84, activation = 'relu'))
  model.add(layers.Dense(18, activation = 'softmax')) 
  
  return model

## Create the model
doodle_model = create_model((X_train.shape[1], X_train.shape[2], 1))
print (doodle_model.summary())

## Compile the model
doodle_model.compile (optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


## Fit the model
history = doodle_model.fit (x = X_train, y = Y_train, epochs = 15, batch_size = 128)
plt.plot(history.history['loss'])

metric_measures = doodle_model.evaluate(x = X_test, y = Y_test)
print ("Loss = " + str(metric_measures[0]))
print ("Test Accuracy = " + str(metric_measures[1]))



## Let us make a prediction with our doodle_model.
ind = 784
img = X_test[ind]
plt.imshow(img.reshape(32, 32), cmap='gray_r')
pred_pobs = doodle_model.predict(np.expand_dims(img, axis=0))
prediction = out_classes[np.argmax(pred_pobs[0, :])]
print(prediction)

doodle_model.save("/Users/ehsanjaveddeveloper/Desktop/doodle_model.h5")
     