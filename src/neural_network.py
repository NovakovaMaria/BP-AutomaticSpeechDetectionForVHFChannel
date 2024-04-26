# author: Maria Novakova
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import main as mp
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import balancedataset as bd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from models import CNN2D, NeuralNet
from sklearn.metrics import precision_recall_curve,average_precision_score, roc_curve, auc
import argparse


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

##########################################################################################

# Hyper-parameters 
input_size = 420
hidden_size1 = 100
hidden_size2 = 50
num_classes = 2
batch_size = 32
num_epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################################################################################

def loaddata(net):
    """data loading

    Args:
        net (bool): net mode

    Returns:
        tensor: tensor datasets
    """
    x, y0, y1 = mp.getData(net)

    x, y0, y1, bonus = bd.balancedataset(x,y0,y1) 


    print(f"Final number of frames: {len(y1)}.")

    mainpart = int(len(x)*0.70)
    smallpart = int(len(x)*0.15)

    # INPUT
    x_train = x[:mainpart]
    x_val = x[mainpart:mainpart+smallpart]
    x_test = x[smallpart+mainpart:]

    # VAD labels
    y0_train = y0[:mainpart]
    y0_val = y0[mainpart:mainpart+smallpart]
    y0_test = y0[mainpart+smallpart:]

    # PTT labels
    y1_train = y1[:mainpart]
    y1_val = y1[mainpart:mainpart+smallpart]
    y1_test = y1[mainpart+smallpart:]

    x_train = torch.tensor(x_train).type(torch.FloatTensor)
    x_test = torch.tensor(x_test).type(torch.FloatTensor)
    x_val = torch.tensor(x_val).type(torch.FloatTensor)


    train_dataset = TensorDataset(x_train, torch.from_numpy(y0_train), torch.from_numpy(y1_train))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, torch.from_numpy(y0_test), torch.from_numpy(y1_test))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(x_val, torch.from_numpy(y0_val), torch.from_numpy(y1_val))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    print(f"Data loaded, trainset: {len(x_train)}, testset: {len(x_test)}, valset: {len(x_val)}")

    return train_dataloader, test_dataloader, val_dataloader,x_test, y0_test, y1_test

##########################################################################################

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.15):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):

        #print(validation_loss < self.min_validation_loss, validation_loss > self.min_validation_loss)

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def evaluate_batch(model, batch_data, batch_labels_0, batch_labels_1, arr):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        # perform a forward pass on the batch using your model
        intput = batch_data.to(device)
        output0, output1 = model(intput)
        output0, output1 = output0.to('cpu'), output1.to('cpu')

        val_loss0 = criterion(output0, batch_labels_0)
        val_loss1 = criterion(output1, batch_labels_1)

        y_true = batch_labels_0.detach().numpy()
        y_prob = output0.detach().numpy()

        y_prob = y_prob[:, 1]

        y_true = np.ravel(y_true)
        y_prob = np.ravel(y_prob)
        fpr, tpr, thresholds1 = roc_curve(y_true, y_prob)
        arr.append([fpr, tpr])


        # compute the predicted labels from the output
        predicted_labels_0 = torch.argmax(output0, dim=1)
        predicted_labels_1 = torch.argmax(output1, dim=1)

        tp = ((predicted_labels_0 == 1) & (batch_labels_0 == 1)).sum()
        fp = ((predicted_labels_0 == 1) & (batch_labels_0 == 0)).sum()
        fn = ((predicted_labels_0 == 0) & (batch_labels_0 == 1)).sum()
        tn = ((predicted_labels_0 == 0) & (batch_labels_0 == 0)).sum()
        total = len(batch_labels_0)

        tp1 = ((predicted_labels_1 == 1) & (batch_labels_1 == 1)).sum()
        fp1 = ((predicted_labels_1 == 1) & (batch_labels_1 == 0)).sum()
        fn1 = ((predicted_labels_1 == 0) & (batch_labels_1 == 1)).sum()
        tn1 = ((predicted_labels_1 == 0) & (batch_labels_1 == 0)).sum()

        # compute the accuracy of the model on the batch
        accuracy0 = (predicted_labels_0 == batch_labels_0).sum().item() / batch_labels_0.size(0)       
        accuracy1 = (predicted_labels_1 == batch_labels_1).sum().item() / batch_labels_1.size(0)  
        acc = accuracy0 + accuracy1
        acc /= 2

    return accuracy0, accuracy1, acc, val_loss0.item(), val_loss1.item(), tp, fp, fn, tn, tp1, fp1, fn1, tn1, total 

    
##########################################################################################


def train(net):

    x_test, y0_test, y1_test, train_dataloader, test_dataloader, val_dataloader =  loaddata(net)

    model = CNN2D()

    if net:
        model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)
    else:
        model = model.to(device)

    model.train()

    learning_rate = 0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    early_stopper = EarlyStopper()


    print("loss=", criterion)
    print("learning rate=", learning_rate)
    print("batch size=", batch_size)

    print(f"Model loaded on device: {device}")
    loss_array = []
    val_array = []
    earlystop = False

    acc = 0
    val_acc = []
    cnt = 0
    val_loss = 0
    arr = []
    counterarc = 0

    with tqdm(total=num_epochs*(len(train_dataloader)), leave=False, unit='clip') as pbar:
        for epoch in range(num_epochs):
            index = 0
            for batch_data, batch_labels_0, batch_labels_1 in train_dataloader:

                input = batch_data.to(device)

                output0, output1 = model(input)
                output0, output1 = output0.to('cpu'), output1.to('cpu')

                loss0 = criterion(output0, batch_labels_0)
                loss1 = criterion(output1, batch_labels_1)

                loss = loss0+loss1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    val_loss = 0
                    for batch_data_val, batch_labels_val_0, batch_labels_val_1  in val_dataloader:
                        accuracy0, accuracy1, acc, \
                        val_loss0, val_loss1, \
                        tp, fp, fn, tn, \
                        tp1, fp1, fn1, tn1, total = evaluate_batch(model, 
                                                                    batch_data_val, 
                                                                    batch_labels_val_0, 
                                                                    batch_labels_val_1, 
                                                                    arr)
                        val_loss += (val_loss0+val_loss1)
                        counterarc+=1

                loss_array.append(loss.item())
                val_array.append(val_loss/((len(val_dataloader))))

                if cnt % 50 == 0:
                    print("Loss", loss.item())
                    print("Loss", val_array[-1], val_loss, val_loss/((len(val_dataloader))))
                    print("Acuraccy on validation set SPEECH", accuracy0*100,"%", "PTT", accuracy1*100,"%", "OVERALL" ,acc*100,"%")

                    #val_loss = 0

                pbar.set_postfix(lossitem=loss.item(), val0=accuracy0*100, val1=accuracy1*100, valloss=val_array[-1])
                pbar.update()

                torch.save(model.state_dict(), 'experiments/lastexp/model.pth')
                cnt += 1

                if early_stopper.early_stop(val_array[-1]):   
                    print("Early stop.")
                    earlystop = True       
                    break

            if earlystop:
                print("Early stop before overfitting.")
                break

    print("Training ended.")

    correct = 0
    sum = 0
    test = []
    total_main = 0
    tp_main, fp_main, fn_main, tn_main = 0,0,0,0
    tp_main1, fp_main1, fn_main1, tn_main1 = 0,0,0,0

    arr = []
    rcl = []
    prcs = []

    cnt = 0
    accuracy = 0
    counterarc = 0

    with torch.no_grad():
       for batch_data_test, batch_labels_test0, batch_labels_test1 in test_dataloader:
            accuracy0, accuracy1, acc, \
            val_loss0, val_loss1, \
            tp, fp, fn, tn, \
            tp1, fp1, fn1, tn1, total = evaluate_batch(model, 
                                                    batch_data_test, 
                                                    batch_labels_test0, 
                                                    batch_labels_test1, 
                                                    arr)

            counterarc+=1

            total_main += total
            tp_main += tp
            fp_main += fp
            fn_main += fn
            tn_main += tn

            tp_main1 += tp1
            fp_main1 += fp1
            fn_main1 += fn1
            tn_main1 += tn1

            cnt += 1
            accuracy += accuracy0#(accuracy0+accuracy1)

            if cnt % 20 == 0 or cnt == 1: 
                total, tp, fp, fn, tn = total_main, tp_main, fp_main, fn_main, tn_main
                tp1, fp1, fn1, tn1 = tp_main1, fp_main1, fn_main1, tn_main1

                precision = tp.float() / (tp + fp)
                recall = tp.float() / (tp + fn)
                f1_score = 2 * (precision * recall) / (precision + recall)

                precision1 = tp1.float() / (tp1 + fp1)
                recall1 = tp1.float() / (tp1 + fn1)
                f1_score1 = 2 * (precision1 * recall1) / (precision1 + recall1)
                rcl.append(recall1)
                prcs.append(precision1)

                print("VAD")
                print("Accuracy", accuracy0)
                print("Precision", precision)
                print("Recall", recall)
                print("Confusion matrix")
                print(tp, fn)
                print(fp, tn)
                print("F1 score", f1_score)

                print("PTT")
                print("Accuracy", accuracy1)
                print("Precision", precision1)
                print("Recall", recall1)
                print("Confusion matrix")
                print(tp1, fp1)
                print(fn1, tn1)
                print("F1 score", f1_score1)

                print("TEST ACCURACY", accuracy/(2*cnt))

                total_main, tp_main, fp_main, fn_main, tp_main1, fp_main1, fn_main1, tn_main, tn_main1 = 0,0,0,0,0,0,0,0,0


    print("Testing ended.")

    import time

    # get the current time in seconds since the epoch
    seconds = time.time()

    torch.save(model.state_dict(), f'{seconds}model.pth')

    x = np.linspace(0, len(loss_array), len(loss_array))
    
    fig, ax = plt.subplots(1,2)
    fig.tight_layout(pad=5.0)

    ax[0].plot(x, loss_array)
    ax[0].set_xlabel("training data")
    ax[0].set_ylabel("training loss")

    ax[1].plot(x, val_array)
    ax[1].set_xlabel("training data")
    ax[1].set_ylabel("validation loss")

    plt.savefig("training.png")
    plt.close()

    out = np.zeros(len(x_test))

    with torch.no_grad():

        input = x_test.to(device)
        #input = input.unsqueeze(1)

        output0, output1 = model(input)
        output0, output1 = output0.to('cpu'), output1.to('cpu')

        y_true = y0_test#.detach().numpy()
        y_prob = output0.detach().numpy()

        y_prob = y_prob[:, 1]

        y_true = np.ravel(y_true)
        y_prob = np.ravel(y_prob)

        fpr, tpr, thresholds1 = roc_curve(y_true, y_prob)
        fpr1, tpr1, _ = roc_curve(y_true, out)

        auc_score = auc(fpr, tpr)
        auc_score1 = auc(fpr1, tpr1)
        print(auc_score, auc_score1)

        plt.figure(figsize=(7,4))
        plt.plot(fpr, tpr, label=f"VAD AUC {round(auc_score,4)}")

        ########################
        y_true = y1_test#.detach().numpy()
        y_prob = output1.detach().numpy()

        y_prob = y_prob[:, 1]

        y_true = np.ravel(y_true)
        y_prob = np.ravel(y_prob)

        fpr, tpr, thresholds1 = roc_curve(y_true, y_prob)
        fpr1, tpr1, _ = roc_curve(y_true, out)

        auc_score = auc(fpr, tpr)
        auc_score1 = auc(fpr1, tpr1)
        print(auc_score, auc_score1)

        plt.figure(figsize=(7,4))
        plt.plot(fpr, tpr, label=f"PTT AUC {auc_score}")
        plt.plot(fpr1, tpr1, label=f"AUC {auc_score1}")

        plt.xlabel(f"False Positive Rate")
        plt.ylabel(f"True Positive Rate")
        plt.legend()
        plt.savefig("ROC.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument(
        '--net', action='store_true',
        help='Provided parameter is for FNN.')
    
    # Read arguments from command line
    args = parser.parse_args()

    net = args.net

    train(net)