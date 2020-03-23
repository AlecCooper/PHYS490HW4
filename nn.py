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

class VAE(nn.Module):

    #Architecture:

    def __init__(self):
        super(VAE, self).__init__()

        # Create the layers of encoder neural net
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3)
        self.pool1 = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=18,kernel_size=3)
        self.pool2 = nn.MaxPool2d((2,2))
        self.fc1_mu = nn.Linear(72, 50)
        self.fc1_sig = nn.Linear(72, 50)

        # Create the layers of the decoder neural net
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, 150)
        self.fc4 = nn.Linear(150,196)
        self.fc5 = nn.Linear(196,196)


    # Encoder neural net
    def encoder(self,x):

        x = func.relu(self.conv1(x))
        x = self.pool1(x)
        x = func.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.reshape(x,(x.size()[0],72))
        mu = func.relu(self.fc1_mu(x))
        sigma = func.relu(self.fc1_sig(x))

        return mu, sigma

    # Repamaramaterize Function
    def reparamaterize(self, mu, sigma):

        epsilon = torch.randn((sigma.size()[0],sigma.size()[0]))
        
        return mu + torch.mm(epsilon, sigma)

    # Decoder neural net
    def decoder(self, z):

        z = func.relu(self.fc2(z))
        z = func.relu(self.fc3(z))
        z = func.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(z))


    # Feedforward function
    def forward(self,x):

        # Encode
        mu, sigma = self.encoder(x)

        # Repamaramaterize
        z = self.reparamaterize(mu, sigma)

        # Decode
        return self.decoder(z), mu, sigma

    # Sample from the latent distrbution
    def sample(self, mu,sigma):

        z = self.reparamaterize(mu, sigma)
        return self.decoder(z)


    # Reset training weights
    def reset(self):

        # Encoder
        self.conv1.reset_parameters()
        self.pool1.reset_parameters()
        self.conv2.reset_parameters()
        self.pool2.reset_parameters()
        self.fc1.reset_parameters()

        # Decoder
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc5.reset_parameters()