# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:07:36 2018

@author: Vincent
"""
# To get the runtime
import time
start_time_ = time.time()

# For file import
from scipy.io.wavfile import read
from os import listdir
from os.path import isfile, join

# For maths
import numpy as np
import matplotlib.pyplot as plt

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


# Return the pitch on each voiced frames
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
            autocorr = plt.xcorr(frames[i], frames[i], maxlags=maxlag)[1]
            
            # Use only the upper side of the autocorrelation to compute de pitch
            upper_side = autocorr[int(len(autocorr)/2):]
            
            # Find the first peak between 60 Hz and 500 Hz
            index_ = np.argmax(upper_side[lag_min:lag_max])
            
            # Unbiase the index of the peak
            index_ += lag_min
            
            # Compute the picth frequency
            pitch.append(sampling_frequency/index_)
            #print(pitch[i])
            
            # Correct the index to display it on the full plot
            index.append(index_+int(len(autocorr)/2))
            
        else:
            # Correct the temporality of the vector
            pitch.append(0);
            index.append(0);
            
    return np.array(pitch, np.float), index


""" Test bench """
"""
import time
start_time_ = time.time()
import numpy as np
import matplotlib.pyplot as plt
form ts_project_s_lib import wav_import_all, normalize, signal_energy, pitch_voiced_autocorr
"""

# Import a set of samples
samples_set, sampling_frequencies, filenames = wav_import_all ("./samples")
number_of_samples = len(samples_set)

# 1. Autocorrelation-Based Pitch Estimation System:
# Normalize the set of samples
print("Normalize samples set")
start_time = time.time()
normalized_set = []
for i in range(number_of_samples):
    normalized_set.append(normalize(samples_set[i]))
print("Done in (" + str(time.time() - start_time) + " seconds)")

# Slice the normalized set of samples in frames
# Slicing parameters
print("Slices the normalized samples set")
start_time = time.time()
widths_arrays = np.ones(len(normalized_set))*30 # ms
shifting_steps_arrays = np.ones(len(normalized_set))*30 # ms

set_of_samples_frames  = []
for i in range(number_of_samples):
    set_of_samples_frames.append(splicer(normalized_set[i], widths_arrays[i], shifting_steps_arrays[i], sampling_frequencies[i]))
print("Done in (" + str(time.time() - start_time) + " seconds)")

"""
# Compute the energy of each frame in each sample (for test purpose only)
print("Compute the energy of each frame")
start_time = time.time()
energy_of_each_frame_in_each_sample = []
for i in range(number_of_samples):
    energy_of_each_frame_in_each_sample.append(signal_energy(set_of_samples_frames[i]))
print("Done in (" + str(time.time() - start_time) + " seconds)")
"""

print("Compute the pitch")
start_time = time.time()
# Compute the pitch of voiced frames of each sample
pitch_of_voiced_frames_in_each_samples = []
for i in range(number_of_samples):
    plt.figure(i+1)
    pitch_of_voiced_frames_in_each_samples.append(pitch_voiced_autocorr(set_of_samples_frames[i], sampling_frequencies[i], 0.3)[0])
    plt.close('all')
print("Done in (" + str(time.time() - start_time) + " seconds)")

#plt.close('all')

# 4. Building a rule-based system
# Compute the mean value of the pitch for each speaker
print("Compute the mean value of the pitch for each speaker")
start_time = time.time()
n_women = 0
n_man = 0
mean_women = 0
mean_man = 0

for i in range(number_of_samples):
    # The filename for women contain a "(2)" in their filename
    if(filenames[i].endswith("(2).wav")):
        n_women += 1
        mean_women += np.sum(pitch_of_voiced_frames_in_each_samples[i])/len(pitch_of_voiced_frames_in_each_samples[i])
    else:
        n_man += 1
        mean_man += np.sum(pitch_of_voiced_frames_in_each_samples[i])/len(pitch_of_voiced_frames_in_each_samples[i])
try:
    mean_women /= n_women
    mean_man /= n_man
except:
    print("Division by zero")

pitch_threshold = (mean_women + mean_man)/2
print("Done in (" + str(time.time() - start_time) + " seconds)")
print("Finished to compute the model in " + str(time.time() - start_time_) + " seconds")

""" 
For the rule-based system, we use the computed mean value of the mean pitch
values for women and men for a know dataset. Then we use the value as a 
threshold. Under the threshold, the speaker is a man else it's a women.
"""
# Test the proposed rule-based system
print("Test the proposed rule-based system on a new database")
start_time = time.time()
speaker_gender = [] # 1 = Women, 0 = Man
speaker_gender_according_to_file = []
error = []


# UT means Under Test 
samples_set_UT, sampling_frequencies_UT, filenames_UT = wav_import_all ("./samples_to_test")
number_of_samples_UT = len(samples_set_UT)

# Normalize the set of samples
print("Normalize samples set under test")
start_time = time.time()
normalized_set_UT = []
for i in range(number_of_samples_UT):
    normalized_set_UT.append(normalize(samples_set_UT[i]))
print("Done in (" + str(time.time() - start_time) + " seconds)")

# Slice the normalized set of samples in frames
# Slicing parameters
print("Slices the normalized samples set under test")
start_time = time.time()
widths_arrays_UT = np.ones(len(normalized_set_UT))*30 # ms
shifting_steps_arrays_UT = np.ones(len(normalized_set_UT))*30 # ms

set_of_samples_frames_UT  = []
for i in range(number_of_samples_UT):
    set_of_samples_frames_UT.append(splicer(normalized_set_UT[i], widths_arrays_UT[i], shifting_steps_arrays_UT[i], sampling_frequencies_UT[i]))
print("Done in (" + str(time.time() - start_time) + " seconds)")

print("Compute the pitch under test")
start_time = time.time()
# Compute the pitch of voiced frames of each sample
pitch_of_voiced_frames_in_each_samples_UT = []
for i in range(number_of_samples_UT):
    plt.figure(i+1)
    pitch_of_voiced_frames_in_each_samples_UT.append(pitch_voiced_autocorr(set_of_samples_frames_UT[i], sampling_frequencies_UT[i], 0.3)[0])
    plt.close('all')
print("Done in (" + str(time.time() - start_time) + " seconds)")

for i in range(number_of_samples_UT):
    # It's a man speaker
    if sum(pitch_of_voiced_frames_in_each_samples_UT[i])/len(pitch_of_voiced_frames_in_each_samples_UT[i]) < pitch_threshold:
        speaker_gender.append(0)
    else: # It's a women speaker
        speaker_gender.append(1)
        
    # Check the speaker gender according to file
    # The filename for women ends with a "(2).wav" in their filename
    if(filenames_UT[i].endswith("(2).wav")):
        speaker_gender_according_to_file.append(1)
    else:
        speaker_gender_according_to_file.append(0)
    
    error.append(int(speaker_gender_according_to_file[i] != speaker_gender[i]))

print("Error rate = " + str(float(sum(error))/float(len(error))))
print("Done in (" + str(time.time() - start_time) + " seconds)")

print("Finished in " + str(time.time() - start_time_) + " seconds")