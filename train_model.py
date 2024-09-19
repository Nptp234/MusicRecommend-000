import numpy as np
import keras, random, cv2, os
from keras import *
import matplotlib.pyplot as plt
from load_data import load_dataset_train, load_dataset_test
import pandas as pd

saved_model = "saved_model"

train_x, train_y, test_x, test_y, n_classes, genre_new = load_dataset_train()

# Expand the dimensions of the image to have a channel dimension. (nx128x128) ==> (nx128x128x1)
# Assuming train_x is a numpy array with shape (9, 128, 128, 3)
train_x_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_x])
train_x = train_x_gray.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)

test_x_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_x])
test_x = test_x_gray.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

# Normalize the matrices.
test_x = test_x/255
train_x = train_x/255

# Create model
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=64, kernel_size=[7,7], kernel_initializer = initializers.he_normal(seed=1), activation="relu", input_shape=(128,128,1)))
# Dim = (122x122x64)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (61x61x64)
model.add(keras.layers.Conv2D(filters=128, kernel_size=[7,7], strides=2, kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (28x28x128)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (14x14x128)
model.add(keras.layers.Conv2D(filters=256, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (12x12x256)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (6x6x256)
model.add(keras.layers.Conv2D(filters=512, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
# Dim = (4x4x512)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.AveragePooling2D(pool_size=[2,2], strides=2))
# Dim = (2x2x512)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
# Dim = (2048)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.6))
model.add(keras.layers.Dense(1024, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (1024)
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(256, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (256)
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(64, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (64)
model.add(keras.layers.Dense(32, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (32)
model.add(keras.layers.Dense(n_classes, activation="softmax", kernel_initializer=initializers.he_normal(seed=1)))
# Dim = (8)

EPOCHS = 20

if not os.path.exists(saved_model):
    os.makedirs(saved_model)
    model.summary()
    keras.utils.plot_model(model, to_file=f"{saved_model}/model_architecture.jpg")
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(), metrics=['accuracy'])
    pd.DataFrame(model.fit(train_x, train_y, epochs=EPOCHS, verbose=1, validation_split=0.1).history).to_csv(f"{saved_model}/training_history.csv")
    model.save(f"{saved_model}/trained_model.h5")
# score = model.evaluate(test_x, test_y, verbose=1)
# print(score)