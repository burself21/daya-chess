import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNN(nn.Module):
    def __init__(self, batch_size, num_hidden_layers=0, hidden_dims=None):
        super(FFNN, self).__init__()
        self.batch_size = batch_size
        # define layers here
        self.input_dim = 69
        if type(hidden_dims) is int:
            self.hidden_dims = [hidden_dims] * num_hidden_layers
        else:
            self.hidden_dims = hidden_dims
        self.num_hidden_layers = len(self.hidden_dims)
        self.linear_fns = nn.ModuleList()
        self.dropout_fns = nn.ModuleList()

        if self.num_hidden_layers > 0:
            self.linear_fns.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
            self.dropout_fns.append(nn.Dropout(0.1))
            for i in range(1, self.num_hidden_layers):
                self.linear_fns.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
                self.dropout_fns.append(nn.Dropout(0.1 + 0.02 * i))
            self.output_fn = nn.Linear(self.hidden_dims[-1], 1)
        else:
            self.output_fn = nn.Linear(self.input_dim, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, X):
        a = X
        for i in range(len(self.linear_fns)):
            #a = self.relu(self.dropout_fns[i](self.linear_fns[i](a)))
            a = self.relu(self.linear_fns[i](a))
        y = self.output_fn(a)
        return y

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.MaxPool1 = nn.MaxPool2d(2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 7, padding=3)
        self.conv4 = nn.Conv2d(256, 64, 3, padding=1)
        self.MaxPool2 = nn.MaxPool2d(2, padding=1)
        self.Dropout1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(3 * 3 * 64, 1)

    def forward(self, X):
        a = F.relu(self.conv1(X))
        #a = self.MaxPool1(F.relu(self.conv1(X)))
        a = F.relu(self.conv2(a))

        a = self.MaxPool1(a)
        #print(a.shape)

        a = F.relu(self.conv3(a))

        a = F.relu(self.conv4(a))

        a = self.MaxPool2(a)

        #print(a.shape)
        a = a.flatten(1)
        #print(a.shape)
        
        #a = F.relu(self.linear1(a))
        y = self.linear1(a)

        return y


