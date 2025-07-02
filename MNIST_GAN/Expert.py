import torch
from torch import nn


class Expert(nn.Module):

    def __init__(self, input_size: int, cuda: str):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 512).to(cuda)
        self.hidden_layer = nn.Linear(512, 256).to(cuda)
        self.output_layer = nn.Linear(256, 1).to(cuda)

    def forward(self, x):
        relu = nn.LeakyReLU(0.2, inplace=True)
        sigmoid = nn.Sigmoid()
        x = relu(self.input_layer(x))
        x = relu(self.hidden_layer(x))
        return sigmoid(self.output_layer(x))
