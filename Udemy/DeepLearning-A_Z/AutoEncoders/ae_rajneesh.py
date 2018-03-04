# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 20:56:39 2017

@author: fractaluser
"""

# impory generic python libraries
import numpy as np
import pandas as pd

# impory scikit-learn specific libraries
import sklearn
from sklearn.model_selection import train_test_split

# impory torch specific libraries
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# read movies and ratings data
movies = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, engine='python')
users = pd.read_csv('./ml-1m/users.dat', sep='::', header=None, engine='python')
ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, engine='python')

# preparing training and test set
training_set = pd.read_csv('./ml-100k/u1.base', sep='\t')
test_set = pd.read_csv('./ml-100k/u1.test', sep='\t')
# convert training set to array for pytorch to consume it
training_set = np.array(training_set, dtype='int')
test_set = np.array(test_set, dtype='int')

# getting number of users and movies
nb_users = max(max(training_set[:, 0]), max(test_set[:, 0]))
nb_movies = max(max(training_set[1]), max(test_set[1]))

'''
# Converting the dataset for autoencoder as users in line and movies rating in columns
def convert(data):
    users_ratings = []
    for id_user in range(1, nb_users+1):
        id_movies = data[data[:,0]==1][:, 1]
        id_ratings = data[data[:,0]==1][:, 2] 
        
        movie_ratings = np.zeros(nb_movies)       
        movie_ratings[id_movies-1] = id_ratings 
           
        users_ratings.append(list(movie_ratings)) 
        
    return users_ratings
        
training_set = convert(training_set)
test_set = convert(test_set)     
'''

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, data.shape[0] + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Convert training and test set to torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Defining the Autoencoder architecture
