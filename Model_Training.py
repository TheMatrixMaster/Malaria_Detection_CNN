#Model Training

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

X = np.load('D:/cell_images/ProcessedData/image_array.npy')
y = np.load('D:/cell_images/ProcessedData/label_array.npy')

# 30% of data goes to testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=100)
# 30% of data goes to validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=len(y_test), random_state=100)

dense_layers = [1]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:
			NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
			print(NAME)

			model = Sequential()

			model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2, 2)))

			for l in range(conv_layer - 1):
				model.add(Conv2D(layer_size, (3, 3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))

			model.add(Flatten())

			for _ in range(dense_layer):
				model.add(Dense(layer_size))
				model.add(Activation('relu'))

			model.add(Dense(1))
			model.add(Activation('sigmoid'))

			tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

			model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

			model.fit(X_train, y_train,
					batch_size=32,  
					epochs=10, 
					validation_data=(X_val, y_val),
					callbacks=[tensorboard])

model.save('64x3-CNN.model')
