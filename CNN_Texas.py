# -*- coding: utf-8 -*-


from __future__ import print_function
import os
import h5py
os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import librosa

import pandas as pd
 
import collections
from collections import deque

from sklearn.preprocessing import LabelEncoder
import scipy
import soundfile as sf

#---------------------------------------------------------------------------------------------------------
#---------------------------------------the noisy_feature-------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#def noisyfile():
#    
#    df = pd.read_csv('PESQ_Eval_Result.csv')
#
#    df.set_index('fileName', inplace=True)
#    lps_mfcc_final={}
#    phase={}
#    count=0
#    for i in range(10):
#    #
#        
#        f=df.index[i]
##        noisy_signal,sr = librosa.load( 'noisySpeech/'+f,sr = 8000)
#        noisy_signal,sr = sf.read( 'noisySpeech/'+f)
#        the_func=librosa.stft(noisy_signal, n_fft= 256, hop_length= 128).T
#  #      time=s.shape[0]/sr
#        mag_frames = np.absolute(the_func)
#        the_angle=np.angle(the_func.T)
#        after_cos=np.cos(the_angle)
#        after_sin=np.sin(the_angle)*1j
#        rec_phase=after_cos+after_sin
#        phase[i]=rec_phase
#        #computing LPS  & mfcc :
#        final=np.zeros((mag_frames.shape[0],155))        
#        mfcc=np.zeros((mag_frames.shape[0],26))
#        
#        mfcc=librosa.feature.mfcc(y=noisy_signal,sr=sr,n_mfcc=26,n_fft=256,hop_length=128).T
##        print('the shape mfcc',mfcc.shape ,'time',time,'at number',i+1)
#        for i2 in range (mag_frames.shape[0]):
#          for j2 in range (mag_frames.shape[1]):
#            if (mag_frames[i2][j2]==0):
#              mag_frames[i2][j2]=1
#            mag_frames[i2][j2]=math.log(np.power(mag_frames[i2][j2],2),math.exp(1))
#        final=np.concatenate((mag_frames,mfcc),1)
#        add_zeros=np.zeros((8,155))
#        final=np.concatenate((final,add_zeros),0)
#        lps_mfcc_final[i]=final
#      #  count=count+final.shape[0]
#        count=count+1
#        print(count)
#
#     #  print('new_matrix',final.shape)
#  #     print(len(phase),len(lps_mfcc_final))
#    
#
#    the_phase=phase[0]  
#    if(len(phase)-1!=0):
#        for uu in range(len(phase)-1):
#            the_phase=np.concatenate((the_phase,phase[uu+1]),0)
#    the_matrix=the_phase.reshape(len(lps_mfcc_final),626,129)
#    circularBuffer = collections.deque(maxlen=9)
#    inputMatrix = np.zeros((9,155))
#    outputMatrix = np.zeros((1,155))
#    for p in range(len(lps_mfcc_final)):
#        the_final_inputt=lps_mfcc_final[p][0:9]
#
#        # Label.append(df.label[p])
#        for mmm in range(9):
#            circularBuffer.append(the_final_inputt[mmm])
#        outputMatrix = circularBuffer.popleft()
#        itr=lps_mfcc_final[p][9]
#        circularBuffer.append(itr)
#        for i in range(len(circularBuffer)):
#            inputMatrix[i]=circularBuffer[i]
#        # Label.append(df.label[p])
#        the_final=np.concatenate((the_final_inputt,inputMatrix),0)
#        for m in range(10,lps_mfcc_final[p].shape[0]):
#            itrr=lps_mfcc_final[p][m]
#            outputMatrix = circularBuffer.popleft()
#            circularBuffer.append(itrr)
#            for i in range(len(circularBuffer)):
#                inputMatrix[i]=circularBuffer[i]
#            # Label.append(df.label[p])
#            the_final=np.concatenate((the_final,inputMatrix),0)
#        hf = h5py.File('F:\\dataset\\test10_noisy\\'+str(p)+'.h5', 'w') #path for folder you willl save in
#        hf.create_dataset('noisy',data=the_final, compression="gzip", compression_opts=9) 
#        hf.close()
#            
#        
#
#    return the_phase
#noisy_phase=noisyfile()
#with open('F:\\dataset\\test10_rere.pickle', 'wb') as f:
#    pickle.dump([noisy_phase], f)


################################################################################################################################################
#hf = h5py.File(str(buffer_number) +'.h5', 'w') 
#g1.create_dataset('phase',data=inputMatrix, compression="gzip", compression_opts=9)
#g2.create_dataset('noisy',data=phase, compression="gzip", compression_opts=9)
#hf.close()



#---------------------------------------------------------------------------------------------------------
#---------------------------------------the clean_feature-------------------------------------------------
#---------------------------------------------------------------------------------------------------------
def cleanfile():
    

    df = pd.read_csv('clean_meta.csv')
 #   df=df[1:]
    df.set_index('fileName', inplace=True)
    lps_mfcc_final={}
 #   Mfcc_feature2=[]
    
    # z = ['000', '00', '0', '']
    count=0
    ccc=0
    for i in range(10):
    #
        
        f=df.index[i]
        s,sr = librosa.load( 'clean/'+f,sr = 8000)
#        s,sr = sf.read( 'clean/'+f)
        
        
        print(sr)
        
        the_func=librosa.stft(s, n_fft= 256, hop_length= 128).T
        mag_frames = np.absolute(the_func)
        #computing LPS  & mfcc :
        final=np.zeros((mag_frames.shape[0],155))        
        mfcc=np.zeros((mag_frames.shape[0],26))
        
        mfcc=librosa.feature.mfcc(y=s,sr=sr,n_mfcc=26,n_fft=256,hop_length=128).T
        for i2 in range (mag_frames.shape[0]):
          for j2 in range (mag_frames.shape[1]):
            if (mag_frames[i2][j2]==0):
                mag_frames[i2][j2]=1
            mag_frames[i2][j2]=math.log(np.power(mag_frames[i2][j2],2),math.exp(1))
        final=np.concatenate((mag_frames,mfcc),1)
        lps_mfcc_final[i]=final
       
#        count=count+final.shape[0]
  #      print('new_matrix',final.shape)
  #      the_input=lps_mfcc_final[0]  
   #     if(len(lps_mfcc_final)-1!=0):
    for u in range(len(lps_mfcc_final)):
        hf = h5py.File('F:\\dataset\\test10_clean\\'+str(u) +'.h5', 'w')  #path for folder you willl save in
        hf.create_dataset('clean',data=lps_mfcc_final[u], compression="gzip", compression_opts=9) 
        hf.close()

      #          the_input=np.concatenate((the_input,lps_mfcc_final[u+1]),0)
            
    return count
numFrames=cleanfile()
#with open('/media/dell1/1.6TBVolume/SOUND/clean4360file.pickle', 'wb') as f:
#    
#    pickle.dump([cleanOutput], f)
