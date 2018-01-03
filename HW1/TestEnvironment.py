# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:41:46 2017

@author: JesuyeDavid
"""

import os
import numpy as np
import matplotlib as pylab
import pandas as pd
import statsmodels
from scipy.misc.pilutil import imread
import tensorflow as tf


#Carrying out feed forward linear regression on
#five images to determine the number
def main():
    result=[]
    weights=np.random.randint(255, size=(784, 1))
    img=readImages()   
    
    for i in img:
        nparray=np.array(i)
        nparray=nparray.reshape(1,784)
        result.append(myNeuron(nparray, weights))
    print(result)
        

def myNeuron(x,w):    
    mulxy=np.dot(x,w)[0][0]
    b=30#bias
    y=mulxy+b
    return(y)

def readImages():
    img=[]
    img.append(imread('1.png'))
    img.append(imread('2.png'))
    img.append(imread('3.png'))
    img.append(imread('4.png'))
    img.append(imread('5.png'))
    return (img)
        

if __name__ == "__main__":
    main()