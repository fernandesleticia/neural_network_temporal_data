## time series prediction
#prever dados de covid para os do proximo dia(last_available_confirmed), usando número de casos atuais(last_available_confirmed), número de mortes(last_available_deaths), new_confirmed, new_deaths
#facebook profit pra uma só coluna?

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

#sequencia de elementos
#seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
#sequencia com dois atributos em cada elemento
seq = [[10,1], [20,1], [30,1], [40,1], [50,1], [60,1], [70,1], [80,1], [90,1]]

def split_sequence_1(sequence, steps):
    X,y = list(), list()
    for i in range(len(sequence)):
        end = i + steps
        if(end>len(sequence)-1):
            break
        seq_x, seq_y = sequence[i:end], sequence[end]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

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

steps = 3
X, y = split_sequence(seq,steps)

#n_features = 1
n_features = 2
# reashape para usar no LSTM, para ter uma matriz de 3 dimensões
#quantidade de elementos, quantidade de features, número de features por sequência
X = X.reshape((X.shape[0],X.shape[1],n_features))

model = tf.keras.models.Sequential()
#criar modelo LSTM de 50 neurônios
model.add(LSTM(50, activation='relu', input_shape=(steps,n_features)))
#modelo com camada densa com 1 neurônio, pra termos 1 elemento de saída
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#200 épocas de treinamento
model.fit(X,y,epochs=200)
#prox = array([70,80,90])
prox = array([[70,1],[80,1],[90,1]])
prox = prox.reshape((1,steps,n_features))
#predizer o próximo elemento
model.predict(prox)

url = 'https://data.brasil.io/dataset/covid19/caso_full.csv.gz'
def download_data():
    r = requests.get(url, allow_redirects=True)
    open('data.csv.gz', 'wb').write(r.content)
    
def unzip_data():
    with gzip.open('data.csv.gz', 'rb') as f_in:
        with open('data.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    
#download_data()
#unzip_data()

data = pd.read_csv('data.csv')
#filtrando apenas casos do estado do ceará
data = data[(data['state'] == 'CE') & data['city'].isna()]
#ordenando por data
data = data.sort_values(['date'])
#extracting columns
data = data[['last_available_confirmed', 'last_available_deaths', 'new_confirmed', 'new_deaths']]
#removing nil values
data = data.loc[~data['last_available_confirmed'].isna()]
data = data.loc[~data['last_available_deaths'].isna()]
data = data.loc[~data['new_confirmed'].isna()]
data = data.loc[~data['new_deaths'].isna()]

data = data.reset_index(drop = True)

