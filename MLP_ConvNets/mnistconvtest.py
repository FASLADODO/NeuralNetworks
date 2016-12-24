# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:16:59 2016

@author: rish
"""

# -*- coding: utf-8 -*-

#%% 

from keras.datasets import mnist #import the datset
from keras.models import Sequential #import the model type
from keras.layers.core import Dense, Dropout, Activation, Flatten #import layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D #import convolution layers
from keras.utils import np_utils

#to plot import matplotlib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#%%

#batch size to train
batch_size = 128
#number of output classes
nb_classes = 10
#number of epochs to train
nb_epoch = 12

#input image dimensions
img_rows, img_cols = 28, 28
#number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
#convolution kernel size
nb_conv = 3

#%%

# the data shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape the data
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test , nb_classes)

i = 4600
plt.imshow(X_train[i , 0] , interpolation = 'nearest')
print("label : ", Y_train[i,:])

#%%

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                          border_mode = 'valid',
                          input_shape=(1, img_rows , img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25)) #regularization

model.add(Flatten()) #1d layer
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

from keras.optimizers import SGD
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)

model.compile(loss = 'categorical_crossentropy',
              optimizer = sgd)
#model.compile(loss='categorical_crossentropy', optimizer='adadelta')
              
def validation_split(train_set_x,train_set_y,val_ratio):
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - val_ratio)))
    valid_set_x = np.asarray([train_set_x[s] for s in sidx[n_train:]])
    valid_set_y = np.asarray([train_set_y[s] for s in sidx[n_train:]])
    train_set_x = np.asarray([train_set_x[s] for s in sidx[:n_train]])
    train_set_y = np.asarray([train_set_y[s] for s in sidx[:n_train]])
    return (valid_set_x, valid_set_y),(train_set_x,train_set_y)
    
### Validation call
(X_val,Y_val),(X_train, Y_train) = validation_split(X_train,Y_train,0.1666667)
              
#%%
          
#training
model.fit(X_train, Y_train, batch_size=batch_size, 
          nb_epoch = nb_epoch, show_accuracy= True,
          verbose=1, validation_data=(X_val, Y_val))

#model.fit(X_train, Y_train, batch_size=batch_size,
 #         nb_epoch = nb_epoch, show_accuracy= True,
  #        verbose=1, validation_split=0.1666667)

#%%

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])