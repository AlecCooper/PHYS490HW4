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
    def __init__(self,file_location,n_test,x_len=14*14):

        # import the data into a numpy array
        x_data = np.genfromtxt(file_location,usecols=range(x_len),dtype=np.float32)
        y_data = np.genfromtxt(file_location,usecols=(x_len),dtype=np.uint8)

        # Scale to between 0 and 1
        x_data = x_data/255.0
        y_data = y_data/255.0

        # Number of data points
        n_data = len(x_data)

        # we must split our y data from a single scalar into a vector
        y_data_vec = np.empty((n_data,5),dtype=np.float32)
        i = 0
        for y in y_data:
            if (y==2):
                y_data_vec[i] = np.array([0,1,0,0,0],dtype=np.float32)
            if (y==4):
                y_data_vec[i] = np.array([0,0,1,0,0],dtype=np.float32)
            if (y==6):
                y_data_vec[i] = np.array([0,0,0,1,0],dtype=np.float32)
            if (y==8):
                y_data_vec[i] = np.array([0,0,0,0,1],dtype=np.float32)
            if (y==0):
                y_data_vec[i] = np.array([1,0,0,0,0],dtype=np.float32)
            i+=1

        # test and train data
        x_train = np.empty((n_data-n_test,1,14,14),dtype=np.float32)
        y_train = np.empty((n_data-n_test,5),dtype=np.float32)
        x_test = np.empty((n_test,1,14,14),dtype=np.float32)
        y_test = np.empty((n_test,5),dtype=np.float32)

        # split into a training and testing set
        indicies = list(range(n_data))
        rand.shuffle(indicies)

        # randomly select our training and testing data
        train_index = indicies[0:n_data-n_test]
        test_index = indicies[-n_test:]

        # add our randomly selected data into the proper arrays
        n = 0
        for i in train_index:
            x_train[n] = np.resize(x_data[i],(14,14))
            y_train[n] = y_data_vec[i]
            n+=1

        n = 0
        for i in test_index:
            x_test[n] = np.resize(x_data[i],(14,14))
            y_test[n] = y_data_vec[i]
            n+=1

        self.x_train = torch.from_numpy(x_train)
        self.x_test = torch.from_numpy(x_test)
        self.y_train = torch.from_numpy(y_train)
        self.y_test = torch.from_numpy(y_test)


        # Enable requires_grad to allow us to use autograd
        self.x_train.requires_grad = True
        

        



