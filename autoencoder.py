# Autoencoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE (nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        # the following returns the encoded vector x in the 1st hidden layer
        x = self.activation(self.fc1(x))
        # likewise for 2nd
        x = self.activation(self.fc2(x))
        # likewise for 3rd
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # reconstructed output vector
        return x
sae = SAE()

# We need the following variables to be used in training the network:
# - criterion for calculating mean squared error
# - optimizer, because all neural networks require optimizers, experiment either Adam or RMS
# for stochastic gradient descent, to update weights at each epoch
criterion = nn.MSELoss()
# lr = learning rate, do some experimentation
# weight_decay, used to reduce lr after every after few epochs to regulate convergence
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay=0.5)

# Training the SAE
nb_epoch = 50
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            # optimize memory + computations
            # use stochastic gradient descent with respect to compute input
            # so given that target and input have same data, we want to reduce
            # computations on target to save memory, hence the below:
            target.require_grad = False
            # here we only deal with future computations of non-zero values, so we
            # isolate part of the data that are 0 (ie movies that user didn't rate)
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            # call backward() on loss - it just tells us in what direction we need
            # to update the different weights; increase or decrease the weight
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
        print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
        
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('Test loss: '+str(test_loss/s))