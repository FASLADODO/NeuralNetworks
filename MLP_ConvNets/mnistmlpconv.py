# -*- coding: utf-8 -*-

import numpy as np

def validation_split(train_set_x,train_set_y,val_ratio):
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - val_ratio)))
    valid_set_x = np.asarray([train_set_x[s] for s in sidx[n_train:]])
    valid_set_y = np.asarray([train_set_y[s] for s in sidx[n_train:]])
    train_set_x = np.asarray([train_set_x[s] for s in sidx[:n_train]])
    train_set_y = np.asarray([train_set_y[s] for s in sidx[:n_train]])
    return (valid_set_x, valid_set_y),(train_set_x,train_set_y)

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
#from keras.utils.visualize_util import plot
from keras.callbacks import Callback,EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = [];
        self.accuracy  = [];
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('acc'))
        
import math
def preprocess(X_train, X_val,X_test,par):
    X_train =  np.concatenate((X_train,X_val),axis=0)
    print (X_train.shape[0])
    X_train -= np.mean(X_train, axis = 0)
    cov = np.dot(X_train.T,X_train)/X_train.shape[0]
    U,S,V = np.linalg.svd(cov)
    #Xrot = np.dot(X_train, U)
    par = int(math.ceil(U.shape[0] *  par))
    print (par)##U.shape
    Xrot = np.dot(X_train, U[:,:par]) # Xrot_reduced becomes [N x 100]
    X_test = np.dot(X_test, U[:,:par])
    ##Xwhite = Xrot / np.sqrt(S + 1e-5)
    X_train = Xrot[0:50000]
    X_val = Xrot[50000:60000]
    print (X_train.shape, X_val.shape)
    return X_train,X_val,X_test
    
def MLP(batch_size,nb_classes,nb_epoch,X_train, Y_train, X_val,Y_val,X_test,Y_test,opt,dp,par):
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)    
    X_val = X_test.reshape(X_val.shape[0], 784)
    if par < 1:
        X_train,X_val, X_test = preprocess(X_train,X_val,X_test, par)
    elif par > 2:
        X_train =  np.concatenate((X_train,X_val),axis=0)
        X_train = X_train[:,::par]
        X_test = X_test[:,::par]
        X_train,X_val =np.split(X_train,[50000,10000])
    print (X_train.shape)
    size = X_train.shape[1];
    ####
    model = Sequential()
    model.add(Dense(256, input_shape=(size,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(1024))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    history = LossHistory()
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=["accuracy"])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
               verbose=0,callbacks=[history,early_stop],
              validation_data=(X_val, Y_test))
    #predicted = model.predict_classes(X_test)
    #con_mat = confusion_matrix(np_utils.categorical_probas_to_classes(Y_test),predicted)
    #print (con_mat)
    score = model.evaluate(X_test, Y_test,
                           show_accuracy=True, verbose=0)
    return score,history


#### Convoltuional Kernel
def CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,nb_conv,X_train, Y_train, X_val,Y_val,X_test,Y_test,opt,dp):
    model = Sequential()
    k = (dp & 12)>>2
    dp = (dp& 3)
    if k==0:
        w1 = 'glorot_uniform'
        w2 = 'glorot_uniform'
    elif k ==1:
        w1 = 'glorot_uniform'
        w2 = 'he_uniform'
    elif k==2:
        w1 = 'he_uniform'
        w2 = 'glorot_uniform'
    else:
        w1 = 'he_uniform'
        w2 = 'he_uniform'   
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,init=w1,
                        border_mode='valid',subsample=(dp,dp),
                        input_shape=(1, img_rows, img_cols)))
    convonet1 = Activation('relu')                   
    model.add(convonet1)
    ##if ((dp&2)>>1) ==1:
      #3  print('dp2')
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,init=w1,subsample=(dp,dp)))
    convonet2 = Activation('relu')
    model.add(convonet2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    ##if dp&1 ==1:
        ##print ('dp1')
    model.add(Dropout(0.5))
    model.add(Flatten())
    d_size = 128 / (dp * dp)
    print (d_size)
    model.add(Dense(d_size,init=w2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,init=w2))
    model.add(Activation('softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=["accuracy"])
    history = LossHistory()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
       #    show_accuracy=True, verbose=1, validation_split=0.2)
          verbose=0,callbacks=[history,early_stop], validation_data=[X_val, Y_val])
    predicted = model.predict_classes(X_test)
    con_mat = confusion_matrix(np_utils.categorical_probas_to_classes(Y_test),predicted)
    model.summary()
    #from models import model_from_json
    #json_string = model.to_json()
    #print json_string
    #config = model.get_config() 
    #print config
    #print (con_mat)
    ##false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predicted)
    #roc_auc = auc(false_positive_rate, true_positive_rate)
    #plt.title('Receiver Operating Characteristic')
    #plt.plot(false_positive_rate, true_positive_rate, 'b'`, label='AUC = %0.2f'% roc_auc)
    #plt.legend(loc='lower right')
    #plt.plot([0,1],[0,1],'r--')
    #plt.xlim([-0.1,1.2])
    #plt.ylim([-0.1,1.2])
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')
    #plt.show()           
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    return score , model,history,con_mat


batch_size = 128
nb_classes = 10
nb_epoch = 200

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
#3e data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
### Validation call
(X_val,y_val),(X_train, y_train) = validation_split(X_train,y_train,0.1666667)
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y_val = np_utils.to_categorical(y_val, nb_classes)

# SGD No momentum No Dropout
from keras.optimizers import SGD
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer='adadelta')
print ('Conv3')
score0,model,history0,conmat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
         nb_conv,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,1)
print('Test score:', score0[0])
print('Test accuracy:', score0[1])

# SGD No momentum Intenal Dropout
from keras.optimizers import SGD
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=False)
#model.compile(loss='categorical_crossentropy', optimizer='adadelta')
print ('Conv5')
score1,model,history1,conmat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,1)
print('Test score:', score1[0])
print('Test accuracy:', score1[1])

opt = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=False)
par = 1
score2,history2 = MLP(batch_size,nb_classes,nb_epoch,X_train, Y_train, X_val,Y_val,X_test,Y_test,opt,3,par)
print('Test score:', score2[0])
print('Test accuracy:', score2[1])

plt.figure(1)
plt.plot(np.arange(0,len(history0.losses),1), history0.losses, 'r', label='ConvNet SGD')
plt.plot(np.arange(0,len(history1.losses),1), history1.losses, 'g', label='ConvNet2 SGD')
plt.plot(np.arange(0,len(history2.losses),1), history2.losses, 'b', label='MLP SGD')
plt.plot(np.arange(0,len(history3.losses),1), history3.losses, 'c', label='MLP2 SGD')
plt.legend(loc='upper right')
plt.title('Learning curve for Training loss')
plt.ylabel('Train Cost Function')
plt.xlabel('Number of batches')
plt.show()      
plt.savefig('train_loss0.png', bbox_inches='tight')   
### Validation loss
plt.figure(2)
plt.plot(np.arange(0,len(history0.val_loss),1), history0.val_loss, 'r', label='ConvNet SGD')
plt.plot(np.arange(0,len(history1.val_loss),1), history1.val_loss, 'g', label='ConvNet2 SGD')
plt.plot(np.arange(0,len(history2.val_loss),1), history2.val_loss, 'b', label='MLP SGD')
plt.plot(np.arange(0,len(history3.val_loss),1), history3.val_loss, 'c', label='MLP2 SGD')
plt.legend(loc='upper right')
plt.title('Learning curve for Validation loss')
plt.ylabel('Validation Cost Function')
plt.xlabel('Number of batches')
plt.show()
plt.savefig('val_loss0.png', bbox_inches='tight')
###Accuracy
plt.figure(3)
plt.plot(np.arange(0,len(history0.accuracy),1), history0.accuracy, 'r', label='ConvNet SGD')
plt.plot(np.arange(0,len(history1.accuracy),1), history1.accuracy, 'g', label='ConvNet2 SGD')
plt.plot(np.arange(0,len(history2.accuracy),1), history2.accuracy, 'b', label='MLP SGD')
plt.plot(np.arange(0,len(history3.accuracy),1), history3.accuracy, 'c', label='MLP2 SGD')
plt.legend(loc='lower right')
plt.title('Learning curve for Accuracy')
plt.ylabel('Accuracy Curve')
plt.xlabel('Number of batches')
plt.show()  
plt.savefig('accuracy0.png', bbox_inches='tight')


## SGD No momentum Intenal Dropout
from keras.optimizers import RMSprop
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#model.compile(loss='categorical_crossentropy', optimizer='adadelta')
print ('SGD with MP1')
score0,model,history0,conmat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
         nb_conv,X_train, Y_train, X_val,Y_val,X_test,Y_test,rms,1)
print('Test score:', score0[0])
print('Test accuracy:', score0[1])

from keras.optimizers import RMSprop
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#model.compile(loss='categorical_crossentropy', optimizer='adadelta')
print ('SGD with dropout & MP1')
score1,model,history1,conmat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,rms,1)
print('Test score:', score1[0])
print('Test accuracy:', score1[1])

opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
par = 1
score2,history2 = MLP(batch_size,nb_classes,nb_epoch,X_train, Y_train, X_val,Y_val,X_test,Y_test,opt,3,par)
print('Test score:', score2[0])
print('Test accuracy:', score2[1])

opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
par = 1
score3,history3 = MLP2(batch_size,nb_classes,nb_epoch,X_train, Y_train, X_val,Y_val,X_test,Y_test,opt,3,par)
print('Test score:', score3[0])
print('Test accuracy:', score3[1])

plt.figure(1)
plt.plot(np.arange(0,len(history0.losses),1), history0.losses, 'r', label='ConvNet RMS')
plt.plot(np.arange(0,len(history1.losses),1), history1.losses, 'g', label='ConvNet2 RMS')
plt.plot(np.arange(0,len(history2.losses),1), history2.losses, 'b', label='MLP RMS')
plt.plot(np.arange(0,len(history3.losses),1), history3.losses, 'c', label='MLP2 RMS')
plt.legend(loc='upper right')
plt.title('Learning curve for Training loss')
plt.ylabel('Train Cost Function')
plt.xlabel('Number of batches')
plt.show()      
plt.savefig('train_loss0.png', bbox_inches='tight')   
### Validation loss
plt.figure(2)
plt.plot(np.arange(0,len(history0.val_loss),1), history0.val_loss, 'r', label='ConvNet RMS')
plt.plot(np.arange(0,len(history1.val_loss),1), history1.val_loss, 'g', label='ConvNet2 RMS')
plt.plot(np.arange(0,len(history2.val_loss),1), history2.val_loss, 'b', label='MLP RMS')
plt.plot(np.arange(0,len(history3.val_loss),1), history3.val_loss, 'c', label='MLP2 RMS')
plt.legend(loc='upper right')
plt.title('Learning curve for Validation loss')
plt.ylabel('Validation Cost Function')
plt.xlabel('Number of batches')
plt.show()
plt.savefig('val_loss0.png', bbox_inches='tight')
###Accuracy
plt.figure(3)
plt.plot(np.arange(0,len(history0.accuracy),1), history0.accuracy, 'r', label='ConvNet RMS')
plt.plot(np.arange(0,len(history1.accuracy),1), history1.accuracy, 'g', label='ConvNet2 RMS')
plt.plot(np.arange(0,len(history2.accuracy),1), history2.accuracy, 'b', label='MLP RMS')
plt.plot(np.arange(0,len(history3.accuracy),1), history3.accuracy, 'c', label='MLP2 RMS')
plt.legend(loc='lower right')
plt.title('Learning curve for Accuracy')
plt.ylabel('Accuracy Curve')
plt.xlabel('Number of batches')
plt.show()  
plt.savefig('accuracy0.png', bbox_inches='tight')


#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=False)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.5s1wu')
#score4,model,history4,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,1)
#print('Test score:', score4[0])
#print('Test accuracy:', score4[1])
#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.7, nesterov=False)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.7s1wu')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,1)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.5s2wu')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,2)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#sgd = SGD(lr=0.2, decay=1e-6, momentum=0.7, nesterov=False)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.7s2wu')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         nb_conv,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,2)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=False)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.5s1wgu')
#score4,model,history4,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,5)
#print('Test score:', score4[0])
#print('Test accuracy:', score4[1])
#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=False)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.5s1whu')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,9)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=True)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.5s1whuhu')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,13)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#sgd = SGD(lr=0.2, decay=1e-6, momentum=0.5, nesterov=False)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.5s1wgu')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         nb_conv,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,6)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=False)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.5s1whu')
#score4,model,history4,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,10)
#print('Test score:', score4[0])
#print('Test accuracy:', score4[1])
#
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5, nesterov=False)
##model.compile(loss='categorical_crossentropy', optimizer='adadelta')
#print ('SGD Momentum l0.5s1whuhu')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,14)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#    
#from keras.optimizers import RMSprop
#rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#print ('RMSprop')
#score5,model,history5,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         nb_conv,X_train, Y_train, X_val,Y_val,X_test,Y_test,rms,1)
#x=np.arange(0,len(history5.losses),1)
#len(history5.val_loss)
##visual_utils.plot(model5,to_file='modelrms.png',show_shapes=True)
#print('Test score:', score5[0])
#print('Test accuracy:', score5[1])
#
#rms = RMSprop(lr=0.001, rho=0.5, epsilon=1e-06)
#print ('RMSprop')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,1)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#rms = RMSprop(lr=0.002, rho=0.9, epsilon=1e-06)
#print ('RMSprop')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,1)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#from keras.optimizers import RMSprop
#rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#print ('RMSprop')
#score5,model,history5,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         nb_conv,X_train, Y_train, X_val,Y_val,X_test,Y_test,rms,2)
#x=np.arange(0,len(history5.losses),1)
#len(history5.val_loss)
##visual_utils.plot(model5,to_file='modelrms.png',show_shapes=True)
#print('Test score:', score5[0])
#print('Test accuracy:', score5[1])
#
#rms = RMSprop(lr=0.001, rho=0.5, epsilon=1e-06)
#print ('RMSprop')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,2)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])
#
#rms = RMSprop(lr=0.002, rho=0.9, epsilon=1e-06)
#print ('RMSprop')
#score41,model,history41,con_mat = CNN (batch_size,nb_classes,nb_epoch,img_rows, img_cols,nb_filters,nb_pool,
#         5,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,2)
#print('Test score:', score41[0])
#print('Test accuracy:', score41[1])


#### Training cost function
#plt.figure(4)
#plt.plot(np.arange(0,len(history5.losses),1), history5.losses, 'b', label='RMS Prop')
#plt.plot(np.arange(0,len(history4.losses),1), history4.losses, 'r', label='SGDM')
#plt.plot(np.arange(0,len(history3.losses),1), history3.losses, 'g', label='SGD')
#plt.legend(loc='upper right')
#plt.title('Learning curve for Training loss')
#plt.ylabel('Train Cost Function')
#plt.xlabel('Number of batches')
#plt.savefig('train_loss1.png', bbox_inches='tight')
#plt.show()         
#### Validation loss
#plt.figure(5)
#x=np.arange(0,len(history5.val_loss),1)
#plt.plot(np.arange(0,len(history5.val_loss),1), history5.val_loss, 'b', label='RMS Prop')
#plt.plot(np.arange(0,len(history4.val_loss),1), history4.val_loss, 'r', label='SGDM')
#plt.plot(np.arange(0,len(history3.val_loss),1), history3.val_loss, 'g', label='SGDg')
#plt.legend(loc='upper right')
#plt.title('Learning curve for Validation loss')
#plt.ylabel('Train Cost Function')
#plt.xlabel('Number of batches')
#plt.savefig('val_loss1.png', bbox_inches='tight')
#plt.show()
####Accuracy
#plt.figure(6)
#x=np.arange(0,len(history5.accuracy),1)
#plt.plot(np.arange(0,len(history5.val_loss),1), history5.accuracy, 'b', label='RMS Prop')
#plt.plot(np.arange(0,len(history4.val_loss),1), history4.accuracy, 'r', label='SGDM')
#plt.plot(np.arange(0,len(history3.val_loss),1), history3.accuracy, 'g', label='SGD')
#plt.legend(loc='lower right')
#plt.title('Learning curve for Accuracy loss')
#plt.ylabel('Accuracy Curve')
#plt.xlabel('Number of batches')
#plt.savefig('accuracy1.png', bbox_inches='tight')
#plt.show()
##plt.plot(history)
## th
##from keras.optimizers import SGD
##sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
##score = MLP(128,nb_classes,20,img_rows, img_cols,
##    nb_filters,nb_pool,nb_conv,X_train, Y_train, X_val,Y_val,X_test,Y_test,sgd,3)
##print('Test score:', score[0])
##print('Test accuracy:', score[1])
#    