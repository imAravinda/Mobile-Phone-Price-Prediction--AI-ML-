# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:43:14 2023

@author: ASUS
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Mobile_Price_Classification-220531-204702.csv')
data.head(5)

X = data.drop('price_range',axis=1)
Y = data['price_range']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,x_test,Y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=8,activation='relu',input_shape=(X_train.shape[1],)), 
    tf.keras.layers.Dense(units=4,activation='relu'),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']    
)

model.fit(X_train,Y_train,epochs=100,batch_size=32,validation_split=0.2)

model.save_weights('mobile_price_weights.h5')