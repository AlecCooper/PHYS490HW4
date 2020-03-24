# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 4
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import json, argparse
import torch.optim as optim
import torch as torch
import torch.nn.functional as func
import numpy as np
from os import path, mkdir
from data_parse import Data
import nn as nn
import matplotlib.pyplot as plt

# Loss function
def loss_func(z,x,mu,sigma, kl_term):

    # BCE Loss
    bce = func.binary_cross_entropy(torch.flatten(z),torch.flatten(x), reduction="sum")

    # KL Loss
    latent_dist = torch.normal(mu,sigma)
    standard_dist = torch.randn_like(sigma)
    kld = func.kl_div(latent_dist,standard_dist,reduction="sum")

    return bce + kl_term*kld

# Main training loop
def train(hyper, num_epochs, results):

    # Create our model
    vae = nn.VAE()

    # Create our optimizer
    optimizer = optim.Adam(vae.parameters(), lr=hyper["learning rate"])

    # Import our data
    print("Importing data.....")
    data = Data(args.d)
    print("Done")

    loss_vals = []

    # Training loop
    for epoch in range(1, num_epochs + 1):

        # Clear our gradient buffer
        optimizer.zero_grad()

        # Clear gradients
        vae.zero_grad()

        # feed our inputs through the net
        output, mu, sigma = vae(data.x_train)

        # Target
        x = data.x_train
        x.requires_grad = False

        # Calculate the loss
        loss = loss_func(output, x, mu, sigma, hyper["kl term"])

        # Graph our progress
        loss_vals.append(loss.item())

        # Backpropagate our loss
        loss.backward()

        optimizer.step()

        # Sample and save
        plot = sample(vae, mu, sigma)
        plot.savefig(results + "/iterations/" + str(epoch) + ".pdf")

        if args.v>=2:
            if not ((epoch + 1) % hyper["display epochs"]):
                print('Epoch [{}/{}]'.format(epoch, num_epochs) +\
                    '\tTraining Loss: {:.4f}'.format(loss))

    # Plot Results
    plt.clf()
    plt.plot(range(num_epochs), loss_vals, label= "Loss", color="blue")
    plt.show()
    plt.savefig(results + "/loss.pdf")

    print("Done!")
    return vae, mu, sigma

# Create samples from our trained model
def sample(vae, mu, sigma):

    data = vae.sample(mu, sigma)
    data = data.detach().numpy()
    data = data[0]
    data = np.reshape(data,(-1,14))
    plt.imshow(data)
    return plt
    
if __name__ == "__main__":

    # Get Command Line Arguments
    parser = argparse.ArgumentParser(description="VAE in PyTorc")
    parser.add_argument("-p",metavar="params/param_file_name.json",type=str, help="Location of paramater file")
    parser.add_argument("-r",metavar="/results_files",type=str,help="Location of directory to store results")
    parser.add_argument("-d",metavar="data/data_file",type=str,help="Location of data file")
    parser.add_argument("-n",metavar="N",type=int,help="Number of files to output")
    parser.add_argument('-v', type=int, default=1, metavar='N', help='verbosity (default: 1)')
    args = parser.parse_args()

    # Check if the results directory exists, otherwise make one
    if not path.exists(args.r):
        mkdir(args.r)
        mkdir(args.r + "/iterations")

    # Hyperparameters from json file
    with open(args.p) as paramfile:
        hyper = json.load(paramfile)

    # Our program loop runs while running is True
    running = True

    # Main program loop
    while running:
        print("Enter t to enter training mode ")
        print("Enter e to exit ")
        print("Enter s to enter sample mode ")
        command = input()

        if command == "t":
            vae, mu, sigma = train(hyper, args.n, args.r)
        elif command == "s":
            sample(vae,mu,sigma)
        elif command == "e":
            running = False
        else:
            print("Please enter a valid command")


