# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 2
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import torch as torch
import torch.nn as nn
import torch.nn.functional as func

class Encoder(nn.Module):

    #Architecture:

    def __init__(self):
        super(Encoder, self).__init__()

        # Create the layers of the neural net
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3)
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=18,kernel_size=3)
        self.pool2 = nn.MaxPool2d((2,2))

    # Feedforward function
    def forward(self,x):

        x = func.relu(self.conv1(x))
        x = self.pool1(x)
        x = func.relu(self.conv2(x))
        x = self.pool2(x)

        return x

    # Reset training weights
    def reset(self):

        self.conv1.reset_parameters()
        self.pool1.reset_parameters()
        self.conv2.reset_parameters()
        self.pool2.reset_parameters()

class Decoder(nn.Module):

    #Architecture:

    def __init__(self):
        super(Decoder, self).__init__()

        print("!")