from tqdm import tqdm
import numpy as np
import os
import librosa
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import argparse
import configparser

def checkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


'''
Structure of files UNTILL NOW
.
├── /data_example
│   ├── /clean_example
│   └── /noisy_example
│
├── /RCED_Model
│   │
│   ├── /npy_files
│   │   ├── /clean_npy_files
│   │   ├── /noisy_npy_files
│   │   └── /phase_npy_files
│   │
│   ├── DataGenerator.py
│   └── DataPreprocessing.py
'
'''

class DataPreprocessing():
    
    def __init__(self,dataset_path,output_path,sampling_rate,nfft,step,nframe,nmels,
                 superImage_size,std_mean_norm,save_files):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.sampling_rate = sampling_rate
        self.nfft = nfft
        self.step = step
        self.nframe = nframe
        self.nmels = nmels
        self.superImage_size = superImage_size
        self.std_mean_norm = std_mean_norm
        self.save_files = save_files
        
    def extractFeatures(self,file):
        
        fmin = 300 # standard number
        
        # fmax = sampling rate / 2 as standard
        fmax = self.sampling_rate/2
        signal, _ = librosa.load(file, sr=self.sampling_rate)
        
        # Short time fourier transform
        signal_stft = librosa.stft(signal, n_fft=self.nfft, hop_length=self.step, center=False)
        
        # get mag and phase of stft 
        signal_stft_magnitude, signal_stft_phase = librosa.magphase(signal_stft.T)
        signal_stft_magnitude = np.where(signal_stft_magnitude == 0,0.00001,signal_stft_magnitude)
        # log power spectra features
        signal_lps = np.log(np.power(signal_stft_magnitude,2))
        
        # mel spectogram features
        signal_mfsc = librosa.amplitude_to_db( 
                librosa.feature.melspectrogram( 
                        y=signal, sr=self.sampling_rate,
                        n_fft=self.nfft, center = False, 
                        hop_length=self.step, n_mels=self.nmels,
                        fmin=fmin, fmax=fmax
                        ).T
            )
        
        # add both features together
        signal_features = np.concatenate((signal_lps,signal_mfsc), axis =1)
        
        return signal_features, signal_stft_phase    

    
    
    def getInput(self,signal_features):
        
        inputImage = np.asarray([signal_features[i:i+self.nframe,:] for i in range(len(signal_features)-self.nframe+1)])
        
        inputMatrix = inputImage.reshape(inputImage.shape[0], inputImage.shape[2], inputImage.shape[1],1)
        
        return inputMatrix

    
    def ImagesNumber_perFile(self,file):
        
        audio_samples = (librosa.get_duration(filename = file)) * self.sampling_rate
        
        frames_number_perFile = int(((audio_samples - self.nfft) // self.step) + 1)
        
        # number of super images generated per file 
        numSuperImages_perFile = (frames_number_perFile // self.superImage_size) - 1  
        
        # to discard the last fractional image of every file
        numImages_perFile = numSuperImages_perFile * self.superImage_size 
        
        return numSuperImages_perFile , numImages_perFile
    
    
    def main(self):
        
        # clean files path
        clean_path = os.path.join(self.dataset_path,'clean')
        if not os.path.exists(clean_path):
            assert False, ("Clean speech data is required")
        
        # noisy files path
        noisy_path = os.path.join(self.dataset_path,'noisy')
        if not os.path.exists(noisy_path):
            assert False, ("Noisy speech data is required")
            
        # npy files path including clean and noisy npy files
        npy_path = os.path.join(self.output_path,'training_npy_files')
        
        # clean npy files path
        clean_npy_path = os.path.join(npy_path,'clean_npy_files')
        
        # noisy npy files path
        noisy_npy_path = os.path.join(npy_path,'noisy_npy_files')
        
        if self.save_files is True:
            checkdir(npy_path)
            checkdir(clean_npy_path)
            checkdir(noisy_npy_path)
        
        # decide the method to use in normalization , std & mean or min & max
        # std & mean is the default
        if self.std_mean_norm is True:
            clean_scaler_name = os.path.join(self.output_path,'clean_mean_std.npy')
            noisy_scaler_name = os.path.join(self.output_path,'noisy_mean_std.npy')
            
            clean_scaler = StandardScaler()
            noisy_scaler = StandardScaler()
        else:
            clean_scaler_name = os.path.join(self.output_path,'clean_min_max.npy')
            noisy_scaler_name = os.path.join(self.output_path,'noisy_min_max.npy')
            
            clean_scaler = MinMaxScaler()
            noisy_scaler = MinMaxScaler()

        
        for root, dirs, files in os.walk(clean_path):  
            for fileNum, file in tqdm(enumerate(files, start=1)):
                
                clean_file = os.path.join(clean_path, file)
                
                # extract the features of the audio file
                file_clean_features, _ = self.extractFeatures(clean_file)
                
                # calculate mean and std partially
                clean_scaler.partial_fit(file_clean_features)
                
                file_clean_features = file_clean_features[8:]
                
                numSuperImages_perFile , numImages_perFile = self.ImagesNumber_perFile(clean_file)
                
                cleanChunks = np.split(file_clean_features[:numImages_perFile],numSuperImages_perFile)
        
                if self.save_files is True:
                    for i in range (numSuperImages_perFile):
                        clean_name = os.path.join(clean_npy_path,'{}_{}'.format(file.split('.')[0],i+1))
                        
                        np.save(clean_name, np.asarray(cleanChunks[i]))
                
        for root, dirs, files in os.walk(noisy_path):  
            for fileNum, file in tqdm(enumerate(files, start=1)):
                
                noisy_file = os.path.join(noisy_path, file)
                
                noisy_features , _ = self.extractFeatures(noisy_file)
                noisy_scaler.partial_fit(noisy_features)
            
                file_noisy_features = self.getInput(noisy_features)
                
                numSuperImages_perFile , numImages_perFile = self.ImagesNumber_perFile(noisy_file)
                
                noisyChunks = np.split(file_noisy_features[:numImages_perFile],numSuperImages_perFile)
                
                if self.save_files is True:
                    for i in range (numSuperImages_perFile):
                        noisy_name = os.path.join(noisy_npy_path,'{}_{}'.format(file.split('.')[0],i+1))
                        
                        
                        np.save(noisy_name, np.asarray(noisyChunks[i]))
                        
        # mean and std normalization method       
        
        if self.std_mean_norm is True:
            clean_first_norm = clean_scaler.mean_
            clean_second_norm = clean_scaler.scale_
            
            noisy_first_norm = noisy_scaler.mean_
            noisy_second_norm = noisy_scaler.scale_
        else:
            
            clean_first_norm = clean_scaler.data_min_
            clean_second_norm = clean_scaler.data_max_
            
            noisy_first_norm = noisy_scaler.data_min_
            noisy_second_norm = noisy_scaler.data_max_
            
        clean_norm = np.column_stack((clean_first_norm,clean_second_norm))
        np.save(clean_scaler_name, clean_norm)
        
        noisy_norm = np.column_stack((noisy_first_norm,noisy_second_norm))
        np.save(noisy_scaler_name, noisy_norm)
        

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description = 'Data Preprocessing')
   
    parser.add_argument('--dataset_path', type = str, required = True ,
                        help = 'the path of dataset')
    parser.add_argument('--output_path', type = str, required = True ,
                        help = 'the path of npy output files')
    parser.add_argument('--sampling_rate', type = int, default = 8000 ,
                        help = 'sampling rate of audio files')
    parser.add_argument('--fft_points', type = int, default = 256 ,
                        help = 'the number of fft points')
    parser.add_argument('--step_points', type = int, default = 128 ,
                        help = 'the number of step points')
    parser.add_argument('--nframe', type = int, default = 9 ,
                        help = 'the number of consecutive frames for each input image')
    parser.add_argument('--mels_number', type = int, default = 26 ,
                        help = 'the number of mel spectogram features')
    parser.add_argument('--superImage_size', type = int, default = 16 ,
                        help = 'super image size to be saved in numpy file to train on it')
    
    parser.add_argument('--std_mean_norm', action = 'store_true',
                        help = 'use the method of std and mean norm')
    parser.add_argument('--save_files', action = 'store_true',
                        help = 'save npy files')
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    configPath = 'features_configurations.cfg'
    
    config['features_specs'] = {
            'training_npyfiles_path' : args.output_path,
            'fft_points' : args.fft_points,
            'nframe' : args.nframe,
            'nmels' : args.mels_number,
            'superImage_size' : args.superImage_size,
            'std_mean_norm' : bool(args.std_mean_norm)
            }
    
    params = {
            'dataset_path' : args.dataset_path,
            'output_path' : args.output_path,
            'sampling_rate' : args.sampling_rate,
            'nfft' : args.fft_points,
            'step' : args.step_points,
            'nframe' : args.nframe,
            'nmels' : args.mels_number,
            'superImage_size' : args.superImage_size,
            'std_mean_norm' : args.std_mean_norm,
            'save_files' : args.save_files
     }
    
    
    with open(configPath, 'w') as configfile:
        config.write(configfile)
    
    data_preprocessing = DataPreprocessing(**params)
    data_preprocessing.main()
    
    
    