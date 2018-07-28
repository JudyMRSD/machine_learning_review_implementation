# from internet ...

import numpy as np
from matplotlib import pyplot as plt

# extra column of -1 for bias in the weight matrix 
X = np.array([
    [-2,4,-1],
    [-3,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([-1,-1,1,1,1])

def svm_sgd(X, Y):

    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000


    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            if (Y[i]*np.dot(X[i], w)) < 1:
                # eta  = learning rate * C 
                # eta * (-2  *(1/epoch) is (1-learning rate )
                w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
            else:
                # eta * (-2  *(1/epoch)) is (1-learning rate )
                w = w + eta * (-2  *(1/epoch)* w)

    return w

w = svm_sgd(X,y)
print(w) # [ 8.  0.  1.]