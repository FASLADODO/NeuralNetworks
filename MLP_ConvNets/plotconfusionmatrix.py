# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:23:26 2016

@author: rish
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len([0,1,2,3,4,5,6,7,8,9]))
    plt.xticks(tick_marks, [0,1,2,3,4,5,6,7,8,9], rotation=45)
    plt.yticks(tick_marks, [0,1,2,3,4,5,6,7,8,9])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print ("Conv5 FCC256 SGDM")
cm = np.array([[977,0,0,0,0,0,2,1,0,0],
[0,1135,0,0,0,0,0,0,0,0],
[1,0,1029,0,0,0,1,0,1,0],
[0,0,0,1005,0,2,0,1,1,1],
[0,0,0,0,977,0,2,0,1,2],
[0,0,0,6,0,883,1,1,0,1],
[2,3,0,0,1,1,950,0,1,0],
[0,3,0,0,0,0,0,1024,0,1],
[1,1,0,0,0,0,1,0,970,1],
[0,0,0,0,5,2,0,4,2,996]])
 
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cm)
print ("Conv5 FCC256 RMS")
cm = np.array([[978,1,0,0,0,0,0,1,0,0],
[0,1135,0,0,0,0,0,0,0,0],
[1,0,1024,1,1,0,0,5,0,0],
[0,0,0,1007,0,2,0,0,1,0],
[0,0,0,0,972,0,2,1,0,7],
[1,0,0,4,0,885,1,1,0,0],
[3,3,0,0,1,2,947,0,2,0],
[0,3,3,0,0,0,0,1021,0,1],
[1,0,2,1,0,2,0,2,964,2],
[1,0,0,1,4,6,0,2,1,994]])
 
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cm)
print ("MLP256 SGD")
cm = np.array([[972,0,2,0,0,1,2,1,2,0],
[0,1125,4,0,0,0,2,1,3,0],
[2,1,1018,1,1,1,2,4,2,0],
[0,0,6,993,0,4,0,4,2,1],
[1,0,2,0,963,0,3,2,2,9],
[2,0,0,6,1,876,2,1,2,2],
[4,2,1,1,2,4,942,0,2,0],
[2,3,13,3,0,0,0,1003,1,3],
[4,0,4,3,4,2,1,5,948,3],
[3,3,0,5,7,3,0,5,2,981]])
 
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cm)
print ("MLP256 SGD")
cm = np.array([[972,0,1,1,0,1,1,2,1,1],
[0,1123,4,0,0,0,2,1,5,0],
[5,0,1013,0,2,1,2,6,3,0],
[0,0,2,993,0,5,0,6,2,2],
[0,0,3,0,963,0,3,3,1,9],
[2,0,0,7,2,870,4,0,4,3],
[4,2,0,1,3,4,944,0,0,0],
[2,3,12,2,0,1,0,1005,1,2],
[7,2,3,2,7,5,1,5,939,3],
[4,2,0,3,6,1,0,5,0,988]])
 
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cm)