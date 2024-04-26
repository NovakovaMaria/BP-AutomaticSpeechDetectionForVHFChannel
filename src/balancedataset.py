# author: Maria Novakova
import torch.nn as nn
import numpy as np
import random


def balancedataset(x, y0, y1):

    random.seed(42)

    dataset = []
    for i in range(len(x)):
        dataset.append([x[i], y0[i], y1[i]])

    x_new = []
    y0_new = []
    y1_new = []

    ptt = []
    speech = []
    nonspeech = []

    ptt_counter = 0

    for i in range(len(dataset)):
        x_new.append(dataset[i][0])
        y0_new.append(dataset[i][1])
        y1_new.append(dataset[i][2])

    for i in range(len(x_new)):
        for j in range(len(x_new[i])):

            if y1_new[i][j] == 1: # THIS IS PTT FRAME
                ptt.append([x_new[i][j], 0,1])
                ptt_counter += 1

            else: # THIS IS NOT PTT FRAME
                if y0_new[i][j] == 1: # SPEECH FRAME
                    speech.append([x_new[i][j], 1,0])
                else: # NON SPEECH FRAME
                    nonspeech.append([x_new[i][j], 0,0])

    final = []
    print("Number of PTT frames:", len(ptt))
    print("Number of speech frames:", len(speech))
    print("Number of non-speech frames:", len(nonspeech))

    ptt = np.array(ptt, dtype='object')

    pos = random.randint(0, len(speech)-1-2*ptt_counter)
    # balancing speech and non-speech
    speech = np.array(speech[pos:pos+2*int(ptt_counter)], dtype='object')
    pos = random.randint(0, len(nonspeech)-1-2*ptt_counter)
    nonspeech = np.array(nonspeech[pos:pos+2*(ptt_counter)], dtype='object')

    for i in range(len(speech)):
        final.append(speech[i])
    
    for i in range(len(nonspeech)):
        final.append(nonspeech[i])

    for i in range(len(ptt)):
        final.append(ptt[i])

    finalx = []
    finaly0 = []
    finaly1 = []

    for i in range(len(final)):
        finalx.append(final[i][0])
        finaly0.append(final[i][1])
        finaly1.append(final[i][2])

    
    print("Speech frames:", len(speech)/(len(speech)+len(nonspeech)), "Non-speech frames:", (len(nonspeech))/(len(speech)+len(nonspeech)))
    print("PTT frames:", len(ptt)/len(final), "PTT frames:", (len(final)-len(ptt))/len(final))

    return np.array(finalx), np.array(finaly0), np.array(finaly1)