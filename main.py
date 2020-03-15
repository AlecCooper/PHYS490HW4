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
import numpy as np
from os import path, mkdir
from data_parse import Data
from nn import Encoder, Decoder
import matplotlib.pyplot as plt

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
        mkdir(args.results)

    # Hyperparameters from json file
    with open(args.p) as paramfile:
        hyper = json.load(paramfile)

    # Create our model
    encoder = Encoder()
    decoder = Decoder()

    print("Importing data.....")
    data = Data(args.d,3000)
    print("Done")

    # Number of training epochs
    num_epochs = hyper["num epochs"]

    # Training loop
    for epoch in range(1, num_epochs + 1):

        print("!")

