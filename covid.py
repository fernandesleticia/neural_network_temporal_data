from numpy import array
from sklearn.model_selection import train_test_split
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

data = pd.read_csv('data.csv')
#filtering only Ceara cases
data = data[(data['state'] == 'CE') & data['city'].isna()]
#sorting by date
data = data.sort_values(['date'])
#extracting columns
data = data[['last_available_confirmed', 'last_available_deaths', 'new_confirmed', 'new_deaths']]
#removing nil values
data = data.loc[~data['last_available_confirmed'].isna()]
data = data.loc[~data['last_available_deaths'].isna()]
data = data.loc[~data['new_confirmed'].isna()]
data = data.loc[~data['new_deaths'].isna()]

data = data.reset_index(drop = True)

#spliting dataset into train and test
X = data[['last_available_deaths','new_confirmed','new_deaths']]
y = data['last_available_confirmed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=9)

#creating a training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
#shuffle and batch the train dataset.
train_dataset = train_dataset.shuffle(len(data)).batch(9)

#creating a test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#batch the test dataset.
test_dataset = train_dataset.batch(9)

#showing the features and target from train dataset
for feat, targ in train_dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))
  
#creating the model with 3 layers
def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='relu', input_shape = (X_train.shape)),
    tf.keras.layers.Dense(1, activation='relu'),
    tf.keras.layers.Dense(1)
  ])
  #mse compute the quantity that a model should seek to minimize during training
  model.compile(optimizer='adam',loss='mse',metrics='mse')
  return model

#compiling the model
model = get_compiled_model()

#15 times to go through the training set
model.fit(train_dataset, epochs=15)

print('Evaluate')
result = model.evaluate(test_dataset)






