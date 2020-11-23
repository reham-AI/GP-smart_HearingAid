
"""
Created on Fri Mar 13 20:24:45 2020

@author: Reham
"""
import os
os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import math
import h5py
import keras
from keras.layers import Dense,  Flatten, Input #new
from keras.layers import Conv2D 
from keras.models import Model  ##new
from keras.callbacks import LearningRateScheduler
from random import shuffle


#=========================================== data generator=============================
total=np.arange(190)
def generate_arrays_from_file():
    
    i=0
    
    while True:
          
            
      # create numpy arrays of input data
      # and labels, from each line in the file
      x1=h5py.File('F:\\dataset\\batch_noisy\\'+str(total[i])+'.h5', 'r')
      x=np.array(x1.get('noisy'))
      print(x.shape)
      x=x.reshape(int(x.shape[0]/9),9,155,1)
      print(x.shape)
      y1=h5py.File('F:\\dataset\\batch_clean\\'+str(total[i])+'.h5', 'r')
      y=np.array(y1.get('clean'))
      y=y.reshape(y.shape[0],155)
      # yield ({'input_1': x1}, {'output': y})
#      print(x.shape,'y',y.shape)
      print('i=',i)
      if (i==9):
        i=0
      else:
        i=i+1
      
      yield [x],[y]

#=================================adaptive learning rate=======================


def step_decay(epoch):
   initial_lrate = 1e-4
   drop = 0.1 
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((epoch)/epochs_drop))
   print('lr',lrate)
   return lrate
lrate = LearningRateScheduler(step_decay)
#callbacks_list = [lrate]

print('learningRate',lrate)

#=================================save for best epochs=======================
class CustomModelCheckpoint(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(total)
        shuffle(total)

        
        if logs['acc'] > 0.75:
#            print('i am here')
            self.model.save('model_rere.h5', overwrite=True)

cbk = CustomModelCheckpoint()


callbacks_list = [lrate , cbk ]






#=================================Model=======================
input_layer=Input(shape=(9,155,1))
firstConv=Conv2D(129, kernel_size=(5, 1),strides=1,use_bias=True,activation='relu',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.05))(input_layer)
secondConv=Conv2D(43, kernel_size=(5, 1),strides=3,use_bias=True,activation='relu',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.05))(firstConv)
flat=Flatten()(secondConv)
fullyConnec_layer=Dense(1024, activation='relu',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.05))(flat)
output_layer=Dense(155, activation='linear',kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0,stddev=0.05))(fullyConnec_layer)
#out_reshape=Reshape((155,1))(output_layer)
#print(noisyInput.shape, cleanOutput.shape)
model=Model(inputs=[input_layer],outputs=[output_layer])
#model.load_weights('model_rere.h5')


model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])



model.fit_generator(generate_arrays_from_file(),

                    steps_per_epoch=190, epochs=20,callbacks=callbacks_list)







