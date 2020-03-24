# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 4
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import numpy as np
import random as rand
import torch as torch

class Data():

    ## parses the data to be fed into our neural network
    ## file_location is the location of the csv file to be read
    ## n_test is the number of samples to set aside for the testing set
    ## note: x data needs to be a uint8 and y data needs to be a ubyte
    def __init__(self,file_location,x_len=14*14):

        # import the data into a numpy array
        x_data = np.genfromtxt(file_location,usecols=range(x_len),dtype=np.float32)

        # Scale to between 0 and 1
        x_data = x_data/255.0

        # Number of data points
        n_data = len(x_data)

        x_train = np.empty((n_data,1,14,14),dtype=np.float32)

        # split into a training and testing set
        indicies = list(range(n_data))
        rand.shuffle(indicies)

        # randomly select our training and testing data
        train_index = indicies[0:n_data]

        # add our randomly selected data into the proper arrays
        n = 0
        for i in train_index:
            x_train[n] = np.resize(x_data[i],(14,14))
            n+=1

        self.x_train = torch.from_numpy(x_train)

        # Enable requires_grad to allow us to use autograd
        self.x_train.requires_grad = True
        

        



