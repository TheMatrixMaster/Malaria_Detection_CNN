#Model Training

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

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


#Model Evaluation

def GetFalseTruePositiveRate(y_true, y_prob, threshold):

    y_predict = np.fromiter([1 if x > threshold else 0 for x in y_prob ], int)
    n_positives = y_true.sum()
    n_negatives = y_true.shape[0] - n_positives
    
    # get n true positives
    n_true_pos = 0
    n_false_pos = 0
    for pred_value,true_value in zip(y_predict,y_true):
        # true positive
        if true_value == 1 and pred_value == 1:
            n_true_pos += 1
        # false positive
        elif true_value == 0 and pred_value == 1:
            n_false_pos += 1
    true_pos_rate = n_true_pos/n_positives
    false_pos_rate = n_false_pos/n_negatives
    return false_pos_rate,true_pos_rate


def MakeConfusionMatrix(y_true,y_prob,threshold):
    confusion_matrix = np.array([[0,0],[0,0]])
    for pred_value,true_value in zip(y_prob,y_true):
        if true_value == 1:
            #true positive
            if pred_value > threshold:
                confusion_matrix[0,0] += 1
            #false negative
            else:
                confusion_matrix[1,0] += 1
        else:
            #false positive
            if pred_value > threshold: 
                 confusion_matrix[0,1] += 1
            #true negative
            else:
                confusion_matrix[1,1] += 1       
    fig = plt.figure(figsize=(5,5))
    ax =  fig.gca()
    sns.heatmap(confusion_matrix,ax=ax,cmap='Blues',annot=True,fmt='g',
               xticklabels = ['Infected','Uninfected'],
               yticklabels=['Infected','Uninfected'])
    ax.set_ylabel('Actual',fontsize=20)
    ax.set_xlabel('Predicted',fontsize=20)
    plt.title('Confusion Matrix',fontsize=24)
    plt.show()


y_predict = model.predict(X_test)
thresholds = np.arange(0.01, 1.01, 0.01)
thresholds = np.append(np.array([0, 0.00001, 0.001]), thresholds)
roc_auc = np.array([GetFalseTruePositiveRate(y_test, y_predict, n) for n in thresholds])
roc_auc = np.sort(roc_auc, axis=0)
roc_auc_value = roc_auc_score(y_test, y_predict)
loss, accuracy = model.evaluate(X_test, y_test)
accuracy = accuracy
loss = loss
text = 'AUC-ROC score = {:.3f}'.format(roc_auc_value)
text += '\nAccuracy = {:.3f}'.format(accuracy * 100)
text += '\nLoss = {:.3f}'.format(loss)

fig = plt.figure(figsize=(7,7))
ax = fig.gca()
ax.set_title('Malaria AUC-ROC Curve',fontsize=28)
ax.set_ylabel('True Positive Rate',fontsize=20)
ax.set_xlabel('False Positive Rate',fontsize=20)
ax.plot(roc_auc[:,0],roc_auc[:,1])
ax.text(s=text,x=0.1, y=0.8,fontsize=20)
plt.show()


MakeConfusionMatrix(y_test,y_predict,0.5)

