# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:26:41 2018

@author: Vincent
"""
__author__ = 'Vincent STRAGIER'

def main():
    """ Test bench """
    # To get the runtime
    import time
    start_time_ = time.time()

    # Import Numpy for the math
    import numpy as np

    # Import the module with the functions for this project
    from ts_project_s_lib_opti import wav_import_all, normalize, splicer, pitch_voiced_autocorr

    # Import a set of samples
    print("Import first set")
    start_time = time.time()
    samples_set, sampling_frequencies, filenames = wav_import_all ("./samples")
    number_of_samples = len(samples_set)
    print("Done in " + str(time.time() - start_time) + " seconds.\n")

    # 1. Autocorrelation-Based Pitch Estimation System:
    # Normalize the set of samples
    print("Normalize samples set")
    start_time = time.time()
    normalized_set = []
    for i in range(number_of_samples):
        normalized_set.append(normalize(samples_set[i]))
    print("Done in " + str(time.time() - start_time) + " seconds.\n")

    # Slice the normalized set of samples in frames
    # Slicing parameters
    print("Slices the normalized samples set")
    start_time = time.time()
    widths_arrays = np.ones(len(normalized_set))*30 # ms
    shifting_steps_arrays = np.ones(len(normalized_set))*30 # ms

    set_of_samples_frames  = []
    for i in range(number_of_samples):
        set_of_samples_frames.append(splicer(normalized_set[i], widths_arrays[i], shifting_steps_arrays[i], sampling_frequencies[i]))
    # del widths_arrays, shifting_steps_arrays
    print("Done in " + str(time.time() - start_time) + " seconds.\n")

    """
    import matplotlib.pyplot as plt
    # Compute the energy of each frame in each sample (for test purpose only)
    print("Compute the energy of each frame")
    start_time = time.time()
    plt.close('all')
    for i in range(number_of_samples):
        plt.figure(i+1)
        y = signal_energy(set_of_samples_frames[i])
        x = np.arange(len(y))/sampling_frequencies[i]
        plt.plot(x, y)
        
    print("Done in " + str(time.time() - start_time) + " seconds.\n")
    # input('PAUSE, press any key to continue')
    """

    print("Compute the pitch")
    start_time = time.time()
    # Compute the pitch of voiced frames of each sample
    pitch_of_voiced_frames_in_each_samples = []
    for i in range(number_of_samples):
        pitch_of_voiced_frames_in_each_samples.append(pitch_voiced_autocorr(set_of_samples_frames[i], sampling_frequencies[i], 0.3)[0])
    # del set_of_samples_frames, sampling_frequencies
    print("Done in " + str(time.time() - start_time) + " seconds.\n")

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
    print("Done in " + str(time.time() - start_time) + " seconds.\n")
    print("Finished to compute the model in " + str(time.time() - start_time_) + " seconds.\n")

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

    if 1:
        # UT means Under Test 
        print("Import samples to test")
        start_time = time.time()
        samples_set_UT, sampling_frequencies_UT, filenames_UT = wav_import_all ("./samples_to_test")
        number_of_samples_UT = len(samples_set_UT)
        print("Done in " + str(time.time() - start_time) + " seconds.\n")
        
        # Normalize the set of samples
        print("Normalize samples set under test")
        start_time = time.time()
        normalized_set_UT = []
        for i in range(number_of_samples_UT):
            normalized_set_UT.append(normalize(samples_set_UT[i]))
        print("Done in " + str(time.time() - start_time) + " seconds.\n")
        
        # Slice the normalized set of samples in frames
        # Slicing parameters
        print("Slices the normalized samples set under test")
        start_time = time.time()
        widths_arrays_UT = np.ones(len(normalized_set_UT))*30 # ms
        shifting_steps_arrays_UT = np.ones(len(normalized_set_UT))*30 # ms
        
        set_of_samples_frames_UT  = []
        for i in range(number_of_samples_UT):
            set_of_samples_frames_UT.append(splicer(normalized_set_UT[i], widths_arrays_UT[i], shifting_steps_arrays_UT[i], sampling_frequencies_UT[i]))
        print("Done in " + str(time.time() - start_time) + " seconds.\n")
        
        print("Compute the pitch under test")
        start_time = time.time()
        # Compute the pitch of voiced frames of each sample
        pitch_of_voiced_frames_in_each_samples_UT = []
        for i in range(number_of_samples_UT):
            pitch_of_voiced_frames_in_each_samples_UT.append(pitch_voiced_autocorr(set_of_samples_frames_UT[i], sampling_frequencies_UT[i], 0.3)[0])
        print("Done in " + str(time.time() - start_time) + " seconds.\n")

    pitch_treshold = 100  # 134.49506116189434

    print("Apply the rule-based system")
    start_time = time.time()
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
    print("Done in " + str(time.time() - start_time) + " seconds.\n")

    print("Finished in " + str(time.time() - start_time_) + " seconds.")

if __name__ == "__main__":
    main()

"""
from ts_project_s_lib import wav_import_all, normalize, splicer
import matplotlib.pyplot as plt
import numpy as np

show = 0 # Show figures

fig_directory = './figures_genareted'

plt.close('all')

# Extract samples, sampling frequencies and filenames from the wave files in the samples directory
samples_set, sampling_frequencies, filenames = wav_import_all ("./samples")

if show:
    for i in range(len(samples_set)):
        fig = plt.figure(i)
        plt.title(filenames[i] + ' non-normalized, time of sampling = ' + str(len(samples_set[i])/float(sampling_frequencies[i])) + ' s')
        plt.plot(np.arange(len(samples_set[i]))/float(sampling_frequencies[i]), samples_set[i])
        plt.ylabel('Amplitude')
        plt.xlabel('Time [s]')
        fig.set_size_inches(16/2, 9/2, forward=True)
        plt.savefig(fig_directory + '/non-normalized/' + filenames[i] + '_non-normalized.svg', bbox_inches='tight', figsize=(16/2.54, 9/2.54))
    
normalized = normalize(samples_set)

if show:
    for i in range(len(samples_set)):
        fig = plt.figure(i+len(samples_set))
        plt.title(filenames[i] + ' normalized, time of sampling = ' + str(len(samples_set[i])/float(sampling_frequencies[i])) + ' s')
        plt.plot(np.arange(len(normalized[i]))/float(sampling_frequencies[i]), normalized[i])
        plt.ylabel('Amplitude')
        plt.xlabel('Time [s]')
        fig.set_size_inches(16/2, 9/2, forward=True)
        plt.savefig(fig_directory + '/normalized/' + filenames[i] + '_normalized.svg', bbox_inches='tight', figsize=(16/2.54, 9/2.54))

plt.show()

widths_arrays = np.ones(len(normalized))*100 # ms
shifting_steps_arrays = np.ones(len(normalized))*100 # ms
windows_arrays = splicer(normalized, widths_arrays, shifting_steps_arrays, sampling_frequencies)

# DUT
"""
