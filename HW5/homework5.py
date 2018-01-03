# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:02:40 2017

@author: JesuyeDavid
"""

import numpy as np
import h5py
import scipy
from scipy import ndimage 
import matplotlib.pyplot as plt
from testCases_v3 import*
#from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward, predict
from dnn_utils_v2 import*

from lr_utils import load_dataset

#%matplotlib inline
plt.rcParams['figure.figsize']=(5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap']='gray'

#%load_ext autoreload
#%autoreload 2

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    
    W1=np.random.randn(n_h, n_x)*0.01
    b1=np.zeros((n_h, 1))
    W2=np.random.randn(n_y, n_h)*0.01 
    b2=np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                   "b1": b1,
                   "W2": W2,
                   "b2": b2}

    return parameters 
    
'''    
parameters = initialize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters={}
    L=len(layer_dims)
    
    for l in range(1, L):
        parameters['W'+str(l)]=np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b'+str(l)]=np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters

'''
parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''


def linear_forward(A, W, b):
    Z=(np.dot(W,A))+b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

'''    
A, W, b = linear_forward_test_case()
Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))
'''

def linear_activation_forward(A_prev, W, b, activation):
    if activation=="sigmoid":
        Z, linear_cache= linear_forward(A_prev, W, b)
        A, activation_cache=sigmoid(Z)
    elif activation=="relu":
        Z, linear_cache=linear_forward(A_prev, W, b)
        A, activation_cache=relu(Z)
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

'''
A_prev, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation
= "sigmoid")
print("With sigmoid: A = " + str(A))
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation
= "relu")
print("With ReLU: A = " + str(A))
'''

def L_model_forward(X, parameters):
    caches=[]
    A=X
    L=len(parameters)//2
    for l in range(1, L):
        A_prev=A
        A, cache= linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
        caches.append(cache)
    
    AL, cache=linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], "sigmoid" )
    caches.append(cache)
    
    assert(AL.shape==(1, X.shape[1]))
    return AL, caches

'''
X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
'''

def compute_cost(AL, Y):
    m=Y.shape[1]
    logprobs=np.multiply(np.log(AL), Y)
    logprobs2=np.multiply((1-Y), np.log(1-AL))
    cost= - np.sum(logprobs+logprobs2)/m
    cost=np.squeeze(cost)
    assert(cost.shape==())
    
    return cost
'''
Y, AL = compute_cost_test_case()
print("cost = " + str(compute_cost(AL, Y)))
'''

def linear_backward(dZ, cache):
    A_prev, W, b=cache
    m=A_prev.shape[1]
    
    dW=np.dot(dZ, A_prev.T)/m
    db=np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev=np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

'''
dZ, linear_cache = linear_backward_test_case()
dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
'''

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache=cache
    if activation=="relu":
        dZ=relu_backward(dA, activation_cache)
        dA_prev, dW, db= linear_backward(dZ, linear_cache)
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db=linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db
    
'''
AL, linear_activation_cache = linear_activation_backward_test_case()
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")
dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
'''

def L_model_backward(AL, Y, caches):
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)
    
    dAL= -(np.divide(Y, AL)-np.divide(1-Y, 1-AL))
    current_cache=linear_activation_backward(dAL, caches[L-1], 'sigmoid')
    grads["dA" +str(L)], grads["db"+str(L)], grads["db"+str(L)]=current_cache 


    for l in reversed(range(L-1)):
        current_cache=linear_activation_backward(grads["dA" +str(l+2)], caches[l], 'relu')
        dA_prev_temp, dW_temp, db_temp=current_cache
        grads["dA" + str(l+1)]=dA_prev_temp
        grads["dW" + str(l + 1)]=dW_temp
        grads["db" + str(l + 1)]=db_temp
        
    return grads

'''
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)
'''


def update_parameters(parameters, grads, learning_rate):
    L=len(parameters)//2
    for l in range(1, L):
        parameters['W'+str(l)]=parameters['W'+str(l)]-(learning_rate*grads['dW'+str(l)])
        parameters['b'+str(l)]=parameters['b'+str(l)]-(learning_rate*grads['db'+str(l)])
    return parameters
'''
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)
print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))
'''



def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs=[]
    parameters=initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
         AL, caches=L_model_forward(X, parameters)
         cost=compute_cost(AL, Y)
         grads=L_model_backward(AL, Y, caches)
         parameters=update_parameters(parameters, grads, learning_rate)
         if print_cost and i % 100 == 0:
             print ("Cost after iteration %i: %f" %(i, cost))
         if print_cost and i % 100 == 0:
             costs.append(cost)

    
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    
    return parameters




        


def main():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    m_train = train_set_x_orig.shape[0] 
    m_test = test_set_x_orig.shape[0] 
    num_px = train_set_x_orig.shape[1]


    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 

    
    train_set_x = train_set_x_flatten/255 
    test_set_x = test_set_x_flatten/255
   
    layers_dims = [12288, 20, 7, 5, 1] # 5-layer model


    parameters = L_layer_model(train_set_x, train_set_y, layers_dims, num_iterations = 2500, print_cost = True)
    predict(train_set_x, train_set_y, parameters)
    predict(test_set_x, test_set_y, parameters)

if __name__ == '__main__':
    main()    
    
