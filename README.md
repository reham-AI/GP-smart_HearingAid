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


