#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 17:40:54 2021

@author: wenxuan_wang
"""

import librosa # python package for music and audio analysis
x , sr = librosa.load("news1.wav", sr=8000) # the audio used in my filter before
print(x.shape, sr)


import matplotlib.pyplot as plt
import librosa.display 
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
plt.title('Time Domain Waveform')

# STFT Transformation

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))   # transfer amplitude to dB
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('STFT Frequency Spectrum Figure')

# Feature Extraction

#Zero Crossing Rate

# Magnify the figure in the unit time
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
plt.title('Amplified Figure using zero crossing rate')

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)
print(sum(zero_crossings))

# Spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound

import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x[:80000], sr=sr)[0]

# Computing the time variable for visualization

frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames, sr=8000)

# Normalising the spectral centroid for visualisation

def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Plotting the Spectral Centroid along the waveform
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x[:80000], sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
plt.title('Spectral Centroid ')

# Spectral Rolloff

spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
plt.title('Spectral Rolloff')

# MFCC Coefficients

mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)

# Displaying  the MFCCs:

plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.title('MFCC Coefficients')

# load data
import librosa
import numpy as np
import os
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
data_set = []
label_set = []
label2id = {genre:i for i,genre in enumerate(genres)}
id2label = {i:genre for i,genre in enumerate(genres)}
print(label2id)
for g in genres:
    print(g)
    for filename in os.listdir(f'./genres/{g}/'):
        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        data_set.append([float(i) for i in to_append.split(" ")])
        label_set.append(label2id[g])

# Create Dataset
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data_set, dtype = float))
y = np_utils.to_categorical(np.array(label_set))

# Segment Train Set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create Model
from keras import models
from keras.layers import Dense, Dropout
def create_model():
    model = models.Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model
model = create_model()

# Compliation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Evaluation
model.fit(X_train, y_train, epochs=50, batch_size=128)
test_loss, test_acc = model.evaluate(X_test,y_test)
print('test_acc: ',test_acc)
