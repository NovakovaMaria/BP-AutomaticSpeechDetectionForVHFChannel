# author: Maria Novakova
import tarfile
import wave
import scipy
import numpy as np
import librosa, librosa.display
import PEEK as pptlib
from tqdm import tqdm
import random

# calculating melspetrogram for each input signal
def computeSpectogram(data, fs, cnt):
    """ data feature extraction

    Args:
        data (array): signal
        fs (int): sampling rate
        cnt (int): for coordinating augmentation

    Returns:
        array: mel sepctrogram
    """
    
    data = data - np.mean(data) 

    data = data / 2**15 


    data = scipy.signal.medfilt(data)
    std = 0.3 * np.std(data)
    noise = np.random.normal(0,std,len(data))

    if cnt % 2:
        data += noise

    if cnt % 11:
        step = random.randint(1,5)
        data = librosa.effects.pitch_shift(data, sr=fs, n_steps=step)

    lenght = data.size / fs * 1000
    frame_lenght = 25.0
    shift = 10.0
    one_sample_cointains = lenght / data.size
    samples_for_one_frame = int(frame_lenght/one_sample_cointains)
    samples_for_shift = int(shift/one_sample_cointains)

    mel_signal = librosa.feature.melspectrogram(y=data, sr=fs, hop_length=samples_for_shift, n_fft=samples_for_one_frame, window=scipy.signal.hamming, n_mels = 20)

    return mel_signal

# merging labels from .cnet file and calculated ptt position
def paddingLabelsAndMerge(data, lenght, pttlabels):
    """_summary_

    Args:
        data (array): labels for speech
        lenght (int): lenght of corresponding wav
        pttlabels (array): labels for push-to-talk 
    Returns:
        array: padded speech labels
        array: labeled ptt as continuous event for 100ms
    """

    padding = [0]*(lenght-data.size)
    labels = data
    if(len(data) < lenght):
        labels = np.concatenate((data, np.array(padding)), axis=0)
    elif(len(data) > lenght):
        labels = data[:lenght]

    labelsPTT = np.zeros(len(labels))

    frames, peak, lastframe, lastpeak = pttlabels

    for i in range(len(frames)):
        labelsPTT[int(frames[i]):int(frames[i])+10] = [1]*10

    return labels, labelsPTT

# calculating labels for VAD from .cnet file
def parseFile(data):
    """partsing the CNET files

    Args:
        data (array): data from CNET

    Returns:
        array: extracted labels
    """

    d = data.readlines()

    labels = []

    beginPrev = 0.00
    durationPrev = 0.00

    for l in d:
        fill = 0
        line = l.decode("utf-8").split()
        begin, duration = float(line[2]), float(line[3])

        # parsing row in .cnet file
        speech = 1 if line[4].find('<eps>') else 0

        # filling missing labels from .cnet file and setting it to nonspeech for each frame = 10ms
        if( beginPrev + durationPrev < begin):
            fill = round((begin - round(beginPrev+durationPrev,2))*100)
            labels.append([0]*fill)

        # length of added labels based on duration in .cnet file
        mul = round(duration*100)

        # adding labels based on .cnet file, for each frame = 10ms
        labels.append([speech]*mul)

        beginPrev = begin
        durationPrev = duration

    labels = np.concatenate(np.array(labels, dtype=object), axis=0)

    return labels

def groupInput(data, net):
    """groupint input to 20,21 chunks of frames

    Args:
        data (array): mel spectrogram
        net (bool): net mode

    Returns:
        array: grouped input
    """
    data = np.transpose(np.array(data)) # 20,x => x,20
    stacked = []

    # stacking frames to 21x20 bulks 
    for i in range(0, len(data)):
        
        if(i >= 10 and i < len(data) - 11): 
            stacked.append(np.array(data[i-10:i+11])) 
        else: 
            arr = np.zeros((21,20)) # 
            if (i<10): # left padding
                indexOfFrame = 10 - i
                arr[indexOfFrame:] = data[:i+11]
            else: # right padding     
                tmp = np.array(data[i-10:i+11])
                arr[:len(tmp)] = tmp
            stacked.append(arr)

    if net: # labels for FNN
        flat = [stacked[k].flatten() for k in range(len(stacked))] 
        flat = np.array(flat)
    else: # labels for CNN
        flat = np.array(stacked)

    return flat


def loaddata():
    """processing tgz dataset
    """
    data = [] 
    info_data = []
    labels = []
    info_labels = []
    diar_labels = []
    pttLabels = []
    cnt = 0
    csvfile = []
    pttLabels1 = []

    # Open the tgz file using the tarfile module
    #with tarfile.open("./datasetATCO2.tar.xz", "r") as tar:
    with tarfile.open("../ATCO_TEST_SET_4H.tgz", "r") as tar:
    # Iterate over all of the wav files in the tgz file
        with tqdm(total=len(tar.getmembers()), leave=False, unit='clip') as pbar:
            for member in tar.getmembers():
                if member.name.endswith(".wav"):
                    # Open the wav file using the wave module
                    with wave.open(tar.extractfile(member), "r") as wav:
                        print("In training:", member.name[:-4])
                        freq = wav.getframerate()
                        signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

                        pttLabels.append(pptlib.PEEKalgorithm(signal, freq, 160))

                        data.append(computeSpectogram(signal, freq,cnt))
                        pttLabels1.append(pptlib.PTTParse(data[-1]))

                        info = [member.name[:-4], signal.size, freq, signal]
                        info = np.array(info, dtype='object')
                        info_data.append(info)

                        cnt+=1
                        #break


                if member.name.endswith(".cnet"):
                    f = tar.extractfile(member)
                    info_labels.append(member.name[:-5])
                    lab = parseFile(f)
                    labels.append(lab)
                
                pbar.set_postfix(file=member.name)
                pbar.update()
            


    return data, labels, info_labels, info_data, pttLabels, diar_labels, pttLabels1
