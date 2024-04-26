# author: Maria Novakova
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.signal as ss
import librosa
import PEEK
from models import NeuralNet, CNN2D
import argparse

########################## functions ################################3

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

    if net: #FNN
        flat = [stacked[k].flatten() for k in range(len(stacked))] 
        flat = np.array(flat)
    else: #CNN
        flat = np.array(stacked)

    return flat

def extract_feature(data, fs):
    """ data feature extraction

    Args:
        data (array): signal
        fs (int): sampling rate

    Returns:
        array: mel sepctrogram
    """

    window = int(fs * 0.025)
    overlap = int(fs * 0.01)

    data = data - np.mean(data) 

    data = data / 2**15 

    mel_signal = librosa.feature.melspectrogram(y=data, sr=fs, hop_length=overlap, n_fft=window, n_mels = 20)

    return mel_signal

########################## funkcie koniec ################################3
########################## model ################################3

def main(net):
    """main run function

    Args:
        net (bool): true for FNN, false for CNN
    """

    samplerate, data = wavfile.read("wavfile/LKPR_RUZYNE_Radar_120_520MHz_20201024_221711.wav")

    data = np.array(data, dtype="float32")

    frames, _,_,_ = PEEK.PEEKalgorithm(data, samplerate, 160)

    if net:
        model_state_dict = torch.load("../saved_models/FNN/MLT/model.pth")
        model = NeuralNet(420, 100, 50, 2)  

    else:
        model_state_dict = torch.load("../saved_models/CNN/MLT/model.pth")
        model = CNN2D()

    model.load_state_dict(model_state_dict)  # load the state dictionary into the model

    # set the model to evaluation mode
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    mel_signal = extract_feature(data, samplerate)
    input = groupInput(mel_signal, net)


    thresholded = []
    thresholded1 = []

    for i in range(len(input)):

        inpt = input[i].astype(np.float32)

        inpt = torch.tensor(input[i]).type(torch.FloatTensor).to(device)

        output0, output1 = model(inpt)

        output0, output1 = output0.to('cpu'), output1.to('cpu')

        thresholded.append(torch.argmax(output0))
        thresholded1.append(torch.argmax(output1))

    correct = 0

    gtptt = np.zeros(int(len(data)/160))

    gtptt[frames[0]:frames[0]+5] = [1]*(5)
    gtptt[frames[1]:frames[1]+5] = [1]*(5)
    gtptt[frames[2]:frames[2]+5] = [1]*(5)

    for i in range(len(gtptt)):
        if thresholded1[i] == gtptt[i]:
            correct+=1

    print("PTT", np.round(correct/len(gtptt), 2)*100)

    gt = np.zeros(int(len(data)/160))

    start = [39, 101, 132, 162, 199, 244, 279, 333, 398, 458, 639, 802 ]
    end = [96, 120, 160, 198, 234, 276, 324, 393, 438, 627, 699, 1062]

    for i in range(len(start)):
        gt[start[i]:end[i]] = [1] * (end[i]-start[i])
    correct = 0
    for i in range(len(gt)):
        if thresholded[i] == gt[i]:
            correct+=1

    print("VAD", np.round(correct/len(gt),2)*100)


    data = data / (2**15)
    t = np.arange(len(data)) / 160

    f, ax = plt.subplots(5, 1, gridspec_kw={'height_ratios': [2,1,1,1,1]}, figsize=(9, 6)) 

    ax[0].plot(t, data, label="Signal")
    ax[1].plot(gt, label="Ground Truth VAD", color="green")
    ax[2].plot(thresholded, label="predicted VAD", color="orange")
    ax[3].plot(gtptt, label="Ground Truth PTT",  color="red") 
    ax[4].plot(thresholded1, label="predicted Push-to-talk", color="purple") 

    ax[-1].set_xlabel("Frame")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Label")
    ax[2].set_ylabel("Label")
    ax[3].set_ylabel("Label")
    ax[4].set_ylabel("Label")


    ax[0].set_title("Audio")
    ax[1].set_title("Ground Truth VAD")
    ax[2].set_title("predicted VAD")
    ax[3].set_title("Ground Truth PTT")
    ax[4].set_title("predicted PTT")

    ax[1].set_yticks([0,1])
    ax[2].set_yticks([0,1])
    ax[3].set_yticks([0,1])
    ax[4].set_yticks([0,1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--net', action='store_true',
        help='Provided parameter is for FNN.')

    args = parser.parse_args()

    net = args.net

    main(net)
