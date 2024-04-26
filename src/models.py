# author: Maria Novakova
import torch
import torch.nn as nn

# Fully connected neural network with two hidden layer, also called FNN
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size1) 
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)  
        self.relu2 = nn.ReLU()
        
        self.l3 = nn.Linear(hidden_size2, num_classes) 
        self.l4 = nn.Linear(hidden_size2, num_classes) 

        self.softmax = nn.Softmax(dim=0) 

        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)
        nn.init.xavier_uniform_(self.l4.weight)

        self.l1.weight.requires_grad = True
        self.l2.weight.requires_grad = True
        self.l3.weight.requires_grad = True
        self.l4.weight.requires_grad = True
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out1 = self.l3(out)
        out2 = self.l4(out)

        return self.softmax(out1), self.softmax(out2)
    
# Convolutional Neural Network
class CNN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 2)
        self.fc3 = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        else:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)

        x = x.view(-1, 64 * 5 * 5)  # flatten for fully connected layers

        x = nn.functional.relu(self.fc1(x))
        x0 = self.softmax(self.fc2(x)).squeeze()
        x1 = self.softmax(self.fc3(x)).squeeze()

        return x0, x1
    