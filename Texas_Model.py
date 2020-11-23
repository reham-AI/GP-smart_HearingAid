#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D

def create_TexasModel():
    # CNN Model architecture
    model = Sequential()
    # Input Layer
    # The first convolution layer
    model.add(Conv2D(129, kernel_size=(5, 1),
                     strides=1,
                     activation='relu',
                     padding='same',use_bias=True,
                     kernel_initializer='TruncatedNormal',
                     bias_initializer='TruncatedNormal',
                     input_shape=(155,9,1)))
    model.add(BatchNormalization()) 
    
    
    # The second convolution layer
    model.add(Conv2D(43, kernel_size=(5, 1),strides=(3,3), activation='relu'
                ,padding='same',use_bias=True,
                 kernel_initializer='TruncatedNormal',
                 bias_initializer='TruncatedNormal'
                 ))
    
    
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    
    
    # Fully Connected Layer
    model.add(Dense(1024, activation='relu'
                 ,use_bias=True, kernel_initializer='TruncatedNormal',
                 bias_initializer='TruncatedNormal'
                 ))
    model.add(BatchNormalization())
    
    # Output Layer
    model.add(Dense(155, activation='linear'
                ,use_bias=True, kernel_initializer='TruncatedNormal',
                 bias_initializer='TruncatedNormal'
                 ))
    model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1))
    
    
    
    print(model.summary())
    
    return model


