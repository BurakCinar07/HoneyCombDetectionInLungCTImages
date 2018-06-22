import numpy as np
from keras.utils import np_utils
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.models import load_model
import datasource as ds

MODEL_PATH = "C:\\Users\\burak\\Desktop\\sinan"

np.random.seed(123)  # for reproducibility
IMAGE_WIDTH = 21
IMAGE_HEIGHT = 21
model_name = 'model.h5'
path = os.path.join(os.getcwd(), 'resources')
path = os.path.join(path, model_name)
model = load_model(path)
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.load_weights(model_path)
'''
def test_frame(img):
    prediction = model.predict(np.expand_dims(img.reshape(IMAGE_WIDTH, IMAGE_WIDTH, 1), axis=0))
    #return prediction


    prediction = np.squeeze(np.asarray(prediction))
    ind = np.argmax(prediction)
    if prediction[ind] < 0.1:
        return 2
    
    return ind
