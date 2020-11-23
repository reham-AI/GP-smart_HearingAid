# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:23:02 2020

@author: MAHA
"""

from __future__ import print_function
import os
import h5py
os.environ["CUDA_VISIBLE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pyaudio
import scipy
#import wave
import os
#import scipy.io.wavfile as wavfile
import numpy as np
import time
import librosa
from Texas_Model import create_TexasModel
import collections
import wave
import sounddevice as sd
from scipy import interpolate
import threading
from multiprocessing import Queue
#===============================================================
#import itertools
#=========================================================
model_root_path='F:\\dataset'
model_weights = os.path.join(model_root_path,'modelWeights.h5')
clean_scaler_name = os.path.join(model_root_path,'clean_mean_std.npy')
noisy_scaler_name = os.path.join(model_root_path,'noisy_mean_std.npy')
#============================================================
# Model Parameters
rfinal=[]
nfft = 256      
step = 128
nframe = 9
sr = 8000
features_number = 155

# Stream Parameters
CHUNK = 1280
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE =8000

clean_mean_std = np.load(clean_scaler_name)
noisy_mean_std = np.load(noisy_scaler_name)

predicted_mean = clean_mean_std[:,0].reshape(1,features_number)
predicted_std = clean_mean_std[:,1].reshape(1,features_number)

noisy_mean = noisy_mean_std[:,0].reshape(features_number,1,1)
noisy_std = noisy_mean_std[:,1].reshape(features_number,1,1)

noisy_mean = np.repeat(noisy_mean,nframe, axis = 1)
noisy_std = np.repeat(noisy_std,nframe, axis = 1)
#=====================circularBuffer Initialization=============
cb = collections.deque(maxlen=9)
for i in range(9):
    cb.append(np.zeros(155))
#===============================================================
def extractFeatures(signal):
    signal_stft = librosa.stft(signal, n_fft=nfft, hop_length=step , center = True)
    signal_stft_magnitude, signal_stft_phase = librosa.magphase(signal_stft.T)
    signal_lps = np.log(np.power(signal_stft_magnitude,2))
    signal_mfsc = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=signal, sr=sr,n_fft=nfft,
                  center = True, hop_length=step, n_mels=26, fmin=300, fmax=sr/2).T)
    signal_features = np.concatenate((signal_lps,signal_mfsc), axis =1)
    return signal_features, signal_stft_phase    
#========================================================================
def getinput(signal_features):
    #first frame
    inputimage = np.zeros((9,155))
    inputimage1 = np.zeros((9,155))
    new_frame=signal_features[0]
    outputMatrix = cb.popleft()
    cb.append(new_frame) #8 zeros + 1st frame
    for tt in range(len(cb)):
        inputimage1[tt]=cb[tt]
    the_final=inputimage1
    for m in range(1,signal_features.shape[0]): 
            new_frame=signal_features[m]
            outputMatrix = cb.popleft()
            cb.append(new_frame)
            for tt in range(len(cb)):
                inputimage[tt]=cb[tt]
           
            the_final=np.append(the_final,inputimage)
    #print(the_final.shape)
    images=((CHUNK-nfft)//step) +3
    finalImages=the_final.reshape(images,155,9,1)
    #print(finalImages.shape)
    return finalImages
#====================================================
def reconstructSignal(outputmat,phasemat):
    reconstructed_mag_phase = np.multiply(outputmat, np.exp(1j*np.angle(phasemat)))
    reconstructed_Signal=librosa.istft(reconstructed_mag_phase, hop_length=128, center = True)   
    return np.asarray(reconstructed_Signal)
#===============================================
model1 = create_TexasModel()
#model=load_model(model_weights)
model1.load_weights(model_weights)
p_speech = pyaudio.PyAudio()
p_desired = pyaudio.PyAudio()
model2 = create_TexasModel() #for desired model
#model=load_model(model_weights)
model2.load_weights(model_weights) #for desired model 

#======================stream for speech==========================
def callback_speech(in_data, frame_count, time_info, flag):
    
    print('speech')
    # using Numpy to convert to array for processing
   
    audio_data = np.frombuffer(in_data, dtype= np.float32) 
    
    # extract the audio features
    audio_features , audio_phase = extractFeatures(audio_data)   
    # Form the Model Input
    audioInput = getinput(audio_features)
    
    # Normalization
    audio_modelInput = np.subtract(audioInput,noisy_mean)/noisy_std
   
    # Model Output
    output = model1.predict(audio_modelInput)
     
    # Denormalization of Model Output
    predicted_output_denorm = np.asarray((output * predicted_std)+ predicted_mean)   
    # Get the first 129 features (LPS)
    predicted_output = np.sqrt(np.exp(predicted_output_denorm[:,:129])) 
    # Reconstruction of Model Output
    result = reconstructSignal(predicted_output.T,audio_phase.T) # reconstruct the output
    rfinal.append(result)
    return  result, pyaudio.paContinue

#======================stream for desired noise ==========================
def callback_desired(in_data, frame_count, time_info, flag):
    print('desired')
    
    # using Numpy to convert to array for processing
   
    audio_data = np.frombuffer(in_data, dtype= np.float32) 
    
    # extract the audio features
    audio_features , audio_phase = extractFeatures(audio_data)   
    # Form the Model Input
    audioInput = getinput(audio_features)
    
    # Normalization
    audio_modelInput = np.subtract(audioInput,noisy_mean)/noisy_std
   
    # Model Output
    output = model2.predict(audio_modelInput)
     
    # Denormalization of Model Output
    predicted_output_denorm = np.asarray((output * predicted_std)+ predicted_mean)   
    # Get the first 129 features (LPS)
    predicted_output = np.sqrt(np.exp(predicted_output_denorm[:,:129])) 
    # Reconstruction of Model Output
    result = reconstructSignal(predicted_output.T,audio_phase.T) # reconstruct the output
    
    return  result, pyaudio.paContinue
#=============================threading=======================================
seconds=10
def thread1():
    stream = p_speech.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback_speech)

    stream.start_stream()
    print("* recording speech")
    #seconds=20
    while stream.is_active():
        time.sleep(seconds)
        stream.stop_stream()
        print("Stream is stopped")

    stream.close()
    
    p_speech.terminate()
    
def thread2():
    stream = p_desired.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback_desired)

    stream.start_stream()
    print("* recording desired")
    #seconds=20
    while stream.is_active():
        time.sleep(seconds)
        stream.stop_stream()
        print("Stream is stopped")

    stream.close()
    
    p_desired.terminate()
    
#====================================start threading============================#
   
if __name__ == "__main__": 
     t1 = threading.Thread(target=thread1)     
     t2 = threading.Thread(target=thread2) 
     # t3 = threading.Thread(target=f3) 
     # t4 = threading.Thread(target=f4)
     # starting thread  
  
     t1.start()
     t2.start()
     #t1.join()
     #t2.join()
     
     # t3.start()
     # t4.start()
    
