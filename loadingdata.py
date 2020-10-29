import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

dataloc = "data/PetImages"
categories = ['Dog', 'Cat']

for category in categories:
    path = os.path.join(dataloc, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        plt.imshow(img_array, cmap='gray')
        #plt.show()  # display!
        break
    break

print(img_array.shape)

# Image has differenty size, lets handle it.
# img_size = 80
# new_array = cv2.resize(img_array, (img_size, img_size))
# plt.imshow(new_array)
# plt.show()

# training data
train_data = []
def create_traindata():
    for category in categories:
        path = os.path.join(dataloc, category)
        class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
            resize_img_array = cv2.resize(img_array, (img_size, img_size))
            train_data.append([resize_img_array, class_num])
        except Exception as e:
            pass


create_traindata()
print(len(train_data))