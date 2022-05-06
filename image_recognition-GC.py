#Program Documentation
"""
Name: Gio Cisneros
HW11-2
Date:5/6/2022
This program demonstrates how to helps recognize images within Deep Learning   
The program uses Keras, SkLearn, Tensorflow, and other methods in to call this data
CREDITS TO JOSEPHLEE ON GITHUB FOR PROVIDING REFERENCE TO THIS CODE

"""



#Imports
import tensorflow as tf
from tensorflow import keras
import keras
import pandas
import sklearn
import matplotlib
from keras.datasets import cifar10
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from skimage.transform import resize
import numpy as np

#Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#Check x and y shape
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

#print image dataset index 0
print(x_train[0])


#print shape label index 0
img = plt.imshow(x_train[0])
print('\n')
print('The label is:', y_train[0])
print('\n')

#print shape label index 1
img = plt.imshow(x_train[1])
print('\n')
print('The label is:', y_train[1])
print('\n')

#convert the label into a set of 10 numbers 
y_train_one_hot = keras.utils.np_utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.np_utils.to_categorical(y_test, 10)

print('\n')
print('The one hot label is:', y_train_one_hot[1])
print('\n')

#Set the values to be between 0 and 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

#Print array
print('\n')
print(x_train[0])
print('\n')

#Set properties of sequential model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#make this cube-like format of neurons into one row, flatten it
model.add(Flatten())

#Add dense and probability
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#summary of the full architecture
print('\n')
model.summary()
print('\n')

#track the accuracy of our model.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#train our model with batch size 32 and 20 epochs
hist = model.fit(x_train, y_train_one_hot, 
           batch_size=32, epochs=20, 
           validation_split=0.2)




#visualize the model training and validation loss as well as training / validation accuracy over the number of epochs
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


#Test Model
model.evaluate(x_test, y_test_one_hot)[1]

#Save model
#model.save('my_cifar10_model.h5')

filePath = input('Please enter the file path to the cat.jpg file:\n ')
my_image = plt.imread(filePath)
#C:\\Users\\Giovanni Cisneros\\OneDrive\\Documents\\School\\Advanced Python\\HW11\\cat.jpg

#Set Cat Properties
my_image_resized = resize(my_image, (32,32,3))
img = plt.imshow(my_image_resized)
probabilities = model.predict(np.array( [my_image_resized,] ))
#print(probabilities)


number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])

