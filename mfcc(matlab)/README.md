# speech-recognition-system

This is about implementing a signal analysis and speech recognition system using MATLAB.

## Features
Speech_recognition_1.m

- audio signal plotting
- speech start and end points detection
- Discrete Fourier Transformation
- signal pre-emphasize
- signal compression(Linear predictve coding)

Speech_recognition_2.m
- feature extraction(MFCC)
- speech recognition using distortion matrix and dynamic programming

## Data
Data used in this project is recorded by human speaking.\
The sound of "1,2,4,5,6" is recorded twice respectively, divided into 2 groups. \
Repeated recording is mainly for speech recognition part, one group already known, one for test to classify what the speech talks.


## Results
Output is saved in ./result directory.\
Includes: signal wave, DFT wave, Mel cepstrum, point detection result, recognition result.
