#Image Preprocessing Training

import numpy as np 
import os
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2


# Display one image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

# Display two images
def display(a, b, title1 = "Parasited", title2 = "Uninfected"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()

#randomly shuffle cell image list and their labels
def reorder(old_list, order):
    new_list = []
    for i in order:
        new_list.append(old_list[i])
    return new_list

#Defining Image Data Files
parasited = "D:/cell_images/Parasitized/"
uninfected = "D:/cell_images/Uninfected/"

all_parasited_paths = []
all_uninfected_paths = []

#Declaring Training Variables and Labels
cell_images = []
cell_labels = []

#Defining Image Characteristics
height = 50
width = 50

#Pull All File Paths From Parasited Image Path
for file in os.listdir(parasited):
	if file.endswith('.png'):
		file_path = os.path.join(parasited, file)
		all_parasited_paths.append(file_path)

for file in os.listdir(uninfected):
	if file.endswith('.png'):
		file_path = os.path.join(uninfected, file)
		all_uninfected_paths.append(file_path)

#Read, Resize and remove noise from Images in File Paths
for file_path in all_parasited_paths:
	img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)						#Read In The Image (Converts to Numpy Array)
	resized_img = cv2.resize(img, (height, width)).astype('float32')		#Resize the image to 50 x 50 and convert the numpy array values to float 32
	resized_img = resized_img / 255											#Scaling down the pixel values from [0-255] to [0-1]
	blurred_img = cv2.GaussianBlur(resized_img, (1, 1), 0)					#Apply Gaussian Blurring to Image
	cell_images.append(blurred_img)											#Append resized images to the cell images list
	cell_labels.append(1)													#Append label for parasited images - 1

example_parasited = cell_images[-1]											#Example parasited image

for file_path in all_uninfected_paths:
	img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)						#Read in the image (Converts to numpy array)
	resized_img = cv2.resize(img, (height, width)).astype('float32')		#Resize the image to 50 x 50 and convert the numpy array values to float 32
	resized_img = resized_img / 255											#//
	blurred_img = cv2.GaussianBlur(resized_img, (1, 1), 0)					#Apply Gaussian Blurring to Image
	cell_images.append(blurred_img)											#//
	cell_labels.append(0)													#//

example_uninfected = cell_images[-1]										#Example uninfected image

display(example_parasited, example_uninfected)								#Display examples


#Random Shuffle Training Data
np.random.seed(seed=42)														#Set Random Seed
indices = np.arange(len(cell_labels))										#Array of indices corresponding to length of cell images
np.random.shuffle(indices)													#Random Shuffle all indices
indices = indices.tolist()													#List the indices

cell_labels = reorder(cell_labels, indices)									#Reorder cell labels
cell_images = reorder(cell_images, indices)									#Reorder cell images

image_array = np.array(cell_images)											#Convert to numpy array
label_array = np.array(cell_labels)											#Convert to numpy array

np.save('image_array.npy', image_array)
np.save('label_array.npy', label_array)
