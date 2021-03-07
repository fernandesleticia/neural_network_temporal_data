from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import tensorflow as tf
import gzip
import shutil
import requests
import os
from os import listdir
from os.path import isfile,join,abspath
import pandas as pd

seq = [[10,1], [20,1], [30,1], [40,1], [50,1], [60,1], [70,1], [80,1], [90,1]]

steps = 3
n_features = 2

def split_sequence(sequence, steps):
    X,y = list(), list()
    for i in range(len(sequence)):
        end = i + steps
        if(end>len(sequence)-1):
            break
        seq_x, seq_y = sequence[i:end], sequence[end][0]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

X, y = split_sequence(seq,steps)

#reashape to use LSTM, to have a matriz of 3 dimensions
#quantidade de elementos, quantidade de features, número de features por sequência
X = X.reshape((X.shape[0],X.shape[1],n_features))

model = tf.keras.models.Sequential()

#createLSTM model with 50 neuron
model.add(LSTM(50, activation='relu', input_shape=(steps,n_features)))

#model with dens layer of 1 neuron, in order to have 1 output element
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

#200 epochs of training
model.fit(X,y,epochs=200)

prox = array([[70,1],[80,1],[90,1]])
prox = prox.reshape((1,steps,n_features))

#predict the next number
model.predict(prox)





