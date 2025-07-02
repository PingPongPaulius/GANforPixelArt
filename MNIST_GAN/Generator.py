import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, input_size: int, output_size: int, cuda: str):
        super().__init__()
        self.input_layer = nn.Linear(input_size, 256).to(cuda)
        self.hidden_layer = nn.Linear(256, 512).to(cuda)
        self.hidden_2_layer = nn.Linear(512, 1024).to(cuda)
        self.output_layer = nn.Linear(1024, output_size).to(cuda)

    def forward(self, x):
        relu = nn.ReLU(True)
        tanh = nn.Tanh()
        x = relu(self.input_layer(x))
        x = relu(self.hidden_layer(x))
        x = relu(self.hidden_2_layer(x))
        return tanh(self.output_layer(x))
