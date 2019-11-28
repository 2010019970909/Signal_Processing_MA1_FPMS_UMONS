# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:07:36 2018

@author: Vincent STRAGIER
"""
__author__ = 'Vincent STRAGIER'

# For file import
from scipy.io.wavfile import read
from os import listdir
from os.path import isfile, join

# For maths
import numpy as np
import matplotlib.mlab as mlab # To use xcorr without pyplot

# Import all the files in a directory
def wav_import_all(directory_path):
    samples_set = []
    sampling_frequencies = []
    filenames = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    
    for i in range(len(filenames)):
        samples = read((str(directory_path)+ "/" + filenames[i]))
        samples_set.append(np.array(samples[1],np.float))
        sampling_frequencies.append(np.array(samples[0],np.float))
        
    return samples_set, sampling_frequencies, filenames

# Normalize samples
def normalize(samples):
    return np.divide(samples, np.amax(np.absolute(samples)))

def splicer(samples, width, shifting_step, frequency):
    frames = []
    
    sampling_frequency = frequency/1000
    numeric_width = abs(int(width*sampling_frequency))
    numeric_shifting_step = abs(int(shifting_step*sampling_frequency))
    test_configuration = numeric_shifting_step-numeric_width
    
    # Determine the number of iteration to apply
    if test_configuration > 0:
        number_of_iterations = (len(samples)-numeric_width)/(test_configuration)
    elif test_configuration == 0:
        number_of_iterations = len(samples)/numeric_width
    else:
        number_of_iterations = -(len(samples)-numeric_width)/(test_configuration)

    # Split in frames
    for j in range(int(number_of_iterations)):
        frames.append(np.split(samples, [j*numeric_shifting_step, j*numeric_shifting_step + numeric_width])[1])
        
    return frames

# Return the energy of each frame
def signal_energy(frames):
    temp = []
    for i in range(len(frames)):
        temp.append(np.sum(np.square(frames[i])))
    return temp

# xcorr without the pyplot submodule ('junk')
def xcorr(x, y, normed=True, detrend=mlab.detrend_none, usevlines=True, maxlags=10, **kwargs):
        Nx = len(x)
        if Nx != len(y):
            raise ValueError('x and y must be equal length')

        x = detrend(np.asarray(x))
        y = detrend(np.asarray(y))

        c = np.correlate(x, y, mode=2)

        if normed:
            c /= np.sqrt(np.dot(x, x) * np.dot(y, y))

        if maxlags is None:
            maxlags = Nx - 1

        if maxlags >= Nx or maxlags < 1:
            raise ValueError('maglags must be None or strictly '
                             'positive < %d' % Nx)

        lags = np.arange(-maxlags, maxlags + 1)
        c = c[Nx - 1 - maxlags:Nx + maxlags]

        return lags, c

# Return the pitch on each voiced frames calculeted with numpy
def pitch_voiced_autocorr(frames, sampling_frequency, threshold = 0.3):
    energy_of_each_frames = signal_energy(frames)
    
    # Convert frenquencies in number of samples
    maxlag = int(sampling_frequency/50.0)
    lag_min = int(sampling_frequency/500.0)
    lag_max = int(sampling_frequency/60.0)
    
    index = []
    pitch = []
    autocorr = 0
    
    for i in range(len(energy_of_each_frames)):
        if energy_of_each_frames[i] >= threshold:
            # Compute the autocorrelation of the voiced frame with maxlag = 50 Hz
            autocorr = xcorr(frames[i], frames[i], maxlags=maxlag)[1]
            
            # Use only the upper side of the autocorrelation to compute de pitch
            upper_side = autocorr[int(len(autocorr)/2):]
            
            # Find the first peak between 60 Hz and 500 Hz
            index_ = np.argmax(upper_side[lag_min:lag_max])
            
            # Unbiase the index of the peak
            index_ += lag_min
            
            # Compute the picth frequency
            pitch.append(sampling_frequency/index_)
            # print(pitch[i])
            
            # Correct the index to display it on the full plot
            index.append(index_+int(len(autocorr)/2))
            
        else:
            # Correct the temporality of the vector
            pitch.append(0)
            index.append(0)
            
    return np.array(pitch, np.float), index

if __name__ == "__main__":
    print("This script is a Python 3 module for the Signal Processing project.")
