# author: Maria Novakova
# algorithm for detecting push-to-talk

import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.signal import hilbert
from scipy.ndimage.filters import gaussian_filter1d
import peaks
import math
from scipy.signal import savgol_filter


def scanSurroundingnegative(peaks_array, lastpeak, audio, sr):
    """checking if the peak is local minimum within its area

    Args:
        peaks_array (array): candidate peaks start position
        lastpeak (array): candidate peaks end position
        audio (array): audio file
        sr (int): sampling rate

    Returns:
        array: selected peaks starting index
        array: selected peaks ending index
    """
    new_peaks = []
    new_last_peaks = []
    stop = int(0.1*sr)

    for i in range(len(peaks_array)):
        index = peaks_array[i]
        while audio[index] < 0:
            index -= 1

        if len(audio[index:peaks_array[i]+stop]) != 0:
            peakmin = np.argmin(audio[index:peaks_array[i]+stop])
            if peakmin + index == peaks_array[i]:
                new_peaks.append(peaks_array[i])
                new_last_peaks.append(lastpeak[i])

    return np.array(new_peaks), np.array(new_last_peaks)

def scanSurroundingpositive(peaks_array, lastpeak, audio, sr):
    """checking if the peak is local maximum within its area

    Args:
        peaks_array (array): candidate peaks start position
        lastpeak (array): candidate peaks end position
        audio (array): audio file
        sr (int): sampling rate

    Returns:
        array: selected peaks starting index
        array: selected peaks ending index
    """
    new_peaks = []
    new_last_peaks = []
    stop = int(0.1*sr)

    for i in range(len(peaks_array)):
        index = peaks_array[i]
        while audio[index] > 0:
            index -= 1
        
        if len(audio[index:peaks_array[i]+stop]) != 0:
            peakmin = np.argmax(audio[index:peaks_array[i]+stop]) 
            #print(peakmin+index, peaks_array[i])
            if peakmin + index == peaks_array[i]:
                new_peaks.append(peaks_array[i])
                new_last_peaks.append(lastpeak[i])

    return np.array(new_peaks), np.array(new_last_peaks)

def mscounternegative(index, audio, sr):
    """function checks for 100 ms duration for negative peaks

    Args:
        index (int): starting point 
        audio (array): audio signal
        sr (int): sampling rate

    Returns:
        bool: bool value if the peak is kept
        lastidex: last index of sample before flip over zero
    """
    flag = True
    i = 0

    stop = int(0.1*sr)

    while index+i < len(audio) and audio[i+index] < 0:
        i+=1
    lastindex = index+i

    if i < stop: 
        flag = False

    return flag, lastindex

def mscounterpositive(index, audio, sr):
    """function checks for 100 ms duration for positive peaks

    Args:
        index (int): starting point 
        audio (array): audio signal
        sr (int): sampling rate

    Returns:
        bool: bool value if the peak is kept
        lastidex: last index of sample before flip over zero
    """
    flag = True
    i = 0
    stop = int(0.1*sr)

    while index+i < len(audio) and audio[i+index] > 0:
        i+=1
        if i >= stop:
            return True, i+1
    lastindex = index+i

    if i < stop:
        flag = False
    return flag, lastindex

def PEEKalgorithm(data, sr, hop_len):
    """PTT algorithm

    Args:
        data (array): input signal
        sr (float): sampling rate
        hop_len (int): size of frame

    Returns:
        array: starting frames
        array: starting samples
        array: ending frames
        array: ending samples
    """

    frames = []

    data = data/2**15
    audio = data

    # detecting local maxima and minima
    peakmax, peak = peaks.peakdet(data, delta=0.2)

    newpeaksnegative = []
    lastpeaknegative = []

    newpeakspositive = []
    lastpeakpositive = []

    # check for 100 ms duration
    for i in peak:
        flag, index = mscounternegative(i, data, sr)
        if flag:
            newpeaksnegative.append(i)
            lastpeaknegative.append(index)

    for i in peakmax:
        flag, index = mscounterpositive(i, data, sr)
        if flag:
            newpeakspositive.append(i)
            lastpeakpositive.append(index)


    peak = np.array(newpeaksnegative)
    lastpeak = np.array(lastpeaknegative)

    # eliminating double pressed signal
    if len(data) != 0:
        peak, lastpeak = scanSurroundingnegative(peak, lastpeak, data, sr)

    peakpositive = np.array(newpeakspositive)
    lastpeakpositive = np.array(lastpeakpositive)

    if len(data) != 0:
        peakpositive, lastpeakpositive = scanSurroundingpositive(peakpositive, lastpeakpositive, data, sr)

    peak = np.sort(np.concatenate((peak, peakpositive), axis=0))
    
    lastpeak = np.sort(np.concatenate((lastpeak, lastpeakpositive), axis=0))
    
    frames = peak//hop_len
    frames = np.array(list((dict.fromkeys(frames))))

    lastframe = lastpeak//hop_len
    lastframe = np.array(list((dict.fromkeys(lastframe))))

    return frames, peak, lastframe, lastpeak



def PTTParse(mel_signal):
    """ first experimental push-to-talk detector

    Args:
        mel_signal (array): spectrogram of a signal

    Returns:
        array: labels
    """

    # transpose spetrogram for indexing in frames
    mel_signal = np.transpose(np.array(mel_signal))
    mel_signal = mel_signal.astype("float64")
    spg = np.abs(mel_signal)
    # calculating power for signal
    power_to_db = librosa.power_to_db(spg, ref=np.max)

    # output labels
    label = [0]*len(mel_signal)

    lowest = -80.0 # finding maximum of power
    lowest_index = 0 # index of the lowest power
    sequenceAFTER = [] # smooth increase
    sequenceBEFORE = [] # rapid decrease
    highest_freq = False 
    ptt = [] # found ptts
    setLow = False

    for i in range(len(power_to_db)):

        # update new maximum
        if(lowest < power_to_db[i][0]):
            lowest = power_to_db[i][0]

            # set all the values
            lowest_index = i
            setLow = True
            sequenceAFTER = []
            sequenceBEFORE = [power_to_db[i-2][0], power_to_db[i-1][0]] # rapid increase

        if setLow: 
            sequenceAFTER.append(power_to_db[i])

        if len(sequenceAFTER) > 64:
            
            check = [np.abs(sequenceAFTER[i][0]) < np.abs(sequenceAFTER[i+1][0]) and np.abs(sequenceAFTER[i][0]) - np.abs(sequenceAFTER[i+1][0]) < 2.0 for i in range(64)] # if smooth increase

            cnt = 0

            for i in range(len(check)): # count how many smooth increases
                if check[i]:
                    cnt += 1
            
            beforecheck = False
            if np.abs(sequenceBEFORE[0]) - np.abs(lowest) > 20.0: # rapid difference in energy
                beforecheck = True

            aftercheck = False
            if cnt/len(check) >= 0.6: # more increases than decreases
                aftercheck = True

            highest_freq = False
            if np.abs(power_to_db[lowest_index][-1]) > 50.0 : # maximum has low power high frequencies
                highest_freq = True

            if aftercheck and beforecheck and highest_freq: # smooth increase and rapid decrease and centralized "negative" signal
                ptt.append(lowest_index)

            # reset all the values
            sequenceBEFORE = []
            sequenceAFTER = []
            lowest = -80.0
            setLow = False
            highest_freq = False

    pttlabels = []

    for i in range(len(mel_signal)):
        if i in ptt:
           pttlabels.append([i-2, i+7])

    return pttlabels

