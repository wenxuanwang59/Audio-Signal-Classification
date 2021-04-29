[audioIn,fs] = audioread('s1A.wav');
coeffs = mfcc(audioIn,fs);

