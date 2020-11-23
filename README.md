# Smart Hearing Aid using Deep Neural Network

Recently after the deeplearning breakthrough in speech enhancement, more progress has been made and the field has become very improvable again and we became able to suppress more noise types. However, in some situations, noise could be more
important than speech like when having a fire alarm in building or siren ambulance while talking ...etc. In situations like these Hearing-impaired people could be put in lethal
situations. So, distinguishing between noise should be taken into consideration before suppressing them.

# feature extraction 

After generating the noisy speech, feature extraction is computed for noisy speech and
clean speech as the purpose of the feature extraction is illustrating the speech signal by a
predetermined number of components of the signal.

We decided to use LPS and MFSC as primary
and secondary features for our model. The details are shown in the next parts starting from
the noisy signal and passing through different stages to get our features to use them in the
speech enhancement task as required.

# CNN architecture

We used CNN in our approach because CNN can perform better than any ANN architecture in
the speech enhancement problem as it utilizes a lesser number of learnable parameters. CNN can
learn higher-order features of the input data throughout the convolutions. It also can accept threedimensional input, unlike other NN. It’s also very suitable for real-time applications which we
need in our approach.


# Noise of interest aware speech enhancement

our hearing aid to be smart and aware of the noise
type. The noise types can be classified to desired noise like a fire alarm or door knock and
undesired noise like babble or café/restaurant noise. Our idea is to make our hearing aid to be
smart and make the system classify the noise existing with speech to desired or undesired and
take the decision whether to make it audible or not.

# First Proposed Architecture
 In the first proposed solution, both speech enhancement and alerting systems are integrated
to add the noise classification smart feature. The goal here is to change our CNN network to be
the noise of interest aware, which means that it has to deal with the noise of interest (desired
noise) just like clean speech. To achieve that, the proposed CNN model is re-trained with
different training set input by making the input to be clean speech added to the desired noise and
undesired noise. The corresponding training output is clean speech added to the desired noise
only which will make the network to be the noise of interest. 

# Second Proposed Architecture
 In the second proposed solution also, both speech enhancement and alerting system are
integrated in the same model to achieve the smart feature and suppress the undesired noise only
as required in the proper way. To achieve that, the proposed CNN model is re-trained with
different training set input by making the input to be clean speech added to undesired noise and
also desired noise added to undesired noise and the corresponding outputs of the network in the
training stage are clean speech and desired noise respectively.

# Third Proposed Architecture
 In the third proposed network, the desired noise enhancement model is added. This newly
added model works in the same way as the speech enhancement model. They have the same
model architecture with the same parameters, they are identical. But the main difference between
them is the input and the target output. In the training stage, the input for both speech
enhancement and desired noise enhancement models are clean speech added to desired and
undesired noise, and the corresponding outputs are clean speech only and desired noise only
respectively. There is a tricky difference between each input of the two models that affect on
signal to noise ratio (SNR) and this should be taken into consideration as to compute SNR, the
signal power should be divided by the noise power. Signal and noise in the two models’ input are
different from each other. For the speech enhancement model, we consider the signal to be the
clean speech and the noise to be desired noise added to undesired noise as the clean speech is the
required model output. For the desired noise enhancement model, we consider the signal to be
the desired noise, and the noise to be clean speech added to undesired noise as the desired noise
is the required model output.

# Real time
To realize a real-time application for our proposed methods, a microphone is used to
capture the audio data where the audio signal is processed and the enhanced signal is heard by
the hearing aid user in real-time with the lowest latency.
The audio input is captured at 48 kHz giving 1536 samples for each 32ms frame. The frames are
then downsampled to 8 kHz producing a frame of size 256 samples that is 32ms in time.
A 256-point FFT is computed for each frame. Only the first half of magnitude spectra is
considered, we get a total of 155 features per frame out of which 129 are LPS and 26 are MFSC.
After generating the features, a circular buffer is created to collect the features from 9 processing
frames to create an input processing image. The circular buffer functions in a first in first out
(FIFO) manner. As shown in fig.1 the frame 10 is the current processing frame. 2-9 are the
preceding frames and frame 1 has left the buffer. When frame 11 enters the buffer, frame 2
leaves the buffer. The 9 processing frames are generated from a chunk of 1280 samples.
 Where the beginning and end of each chunk is padded with zeros to conserve continuity of the
signal. For signal reconstruction, LPS and the phase of the noisy input frame are utilized.
