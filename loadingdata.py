from util import *

# Fix corrupted img
import sys
# sys.exit(int(fix_corrupted_img.main(sys.argv,  "data/PetImages/Cat") or 0))
# sys.exit(int(fix_corrupted_img.main(sys.argv,  "data/PetImages/Dog") or 0))

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

# Image has differenty size, lets handle it.
img_size = 80
# training data
training_data = []

def create_training_data():
    for category in categories:

        path = os.path.join(dataloc,category)  # create path to dogs and cats
        class_num = categories.index(category)  # get class  (0 or a 1). 0=dog 1=cat

        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (img_size, img_size))  
                training_data.append([new_array, class_num])  
            except Exception as e:  # cleaning
                pass

create_training_data()
print(len(training_data))           

# # randomize data
random.shuffle(training_data)
    
# # Sampling data
X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)
print(X[0].reshape(-1, img_size, img_size, 1))
X = np.array(X).reshape(-1, img_size, img_size, 1)


# # save transformed data
pickle_out = open("data/X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("data/y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()