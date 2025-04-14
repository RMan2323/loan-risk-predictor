import torch
import torch.nn as nn

class DoubleLayer(nn.Module):
    def __init__(self, input_size, hidden1, hidden2):
        super(DoubleLayer, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, 1)  # Single output for binary classification

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x  # For BCEWithLogitsLoss, no Sigmoid is needed here
