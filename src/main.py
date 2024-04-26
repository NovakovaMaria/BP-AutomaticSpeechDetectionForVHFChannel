# author: Maria Novakova
import dataprocessing as dp
import argparse


def getData(net):
    x, y0, y1 = main(net)
    return x, y0, y1

def main(net):
    data, labels, info_labels, info_data, pttLabels, diar_labels, oldptt = dp.loaddata()

    newLabels = []
    newData = []
    newinfoData = []
    newPTT = []
    
    print("All frames loaded.")
    
    for i in range(len(info_data)):
        try:
            newLabels.append(labels[info_labels.index(info_data[i][0])])
            newData.append(data[i])
            newinfoData.append(info_data[i])
            newPTT.append(pttLabels[i])
        except:
            print("Skipped:", info_data[i][0])       
            continue

    for i in range(len(newData)):
        lenght = newData[i][0].size
        newLabels[i], newPTT[i] = dp.paddingLabelsAndMerge(newLabels[i], lenght, newPTT[i])
        newData[i] = dp.groupInput(newData[i], net)

    return newData, newLabels, newPTT

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument(
        '--net', action='store_true',
        help='Provided parameter is for FNN.')
    
    # Read arguments from command line
    args = parser.parse_args()
    net = args.net

   #main(net)