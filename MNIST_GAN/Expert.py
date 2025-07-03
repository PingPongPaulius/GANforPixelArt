import torch
from torch import nn
import torch.nn.functional as F


class Expert(nn.Module):

    def __init__(self, input_size: int, cuda: str):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 512).to(cuda)
        self.hidden_layer = nn.Linear(512, 256).to(cuda)
        self.hidden_layer_2 = nn.Linear(256, 128).to(cuda)
        self.output_layer = nn.Linear(128, 1).to(cuda)

    def forward(self, x):
        x = F.leaky_relu(self.input_layer(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.hidden_layer(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.hidden_layer_2(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.output_layer(x))
