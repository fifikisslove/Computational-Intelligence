#!/usr/bin/env python
__author__ = 'bellec,subramoney'

import numpy as np

import toolbox


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param x:
    :param y:
    :return:
    """
    N, n = x.shape
    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture
    # - Hint: use the logistic function sig imported from the file toolbox

    z = x.dot(theta) #Reduce high-dimensional input to a scalar
    h = toolbox.sig(z) #Apply logistic function
    #cost calculation with matrix calculation - gives some 0 multiplication with case 3
    #y1 = -np.log(h) * y
    #y0 = -np.log(1. - h) * (np.invert(y))
    #costs = y0 + y1
    #c=  (1. / N) * np.sum(costs)
    #print(type(c))
    #return  (1. / N) * (-(np.sum(y.dot(np.log(h)) + (1-y).dot(np.log(1.-h)))))

    #cost calculation with lign by lign calculation
    c = 0
    for i in range(0, N):
        if y[i]:
            c -= np.log(h[i])
        else:
            c -= np.log(1. - h[i])
    #print("b" + " " + str(type(c)) + " " + str(c))
    #print("b" + " " + str(type(np.float64(c))) + " " + str(c))
    #case 2 :b <class 'numpy.float64'>
    #case 1 :same
    #case 3 :numpy.ndarray

    return np.float64((1 / N) * c)



    # END TODO
    ###########

def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta:
    :param x:
    :param y:
    :return:
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    z = np.dot(x, theta) #Reduce high-dimensional input to a scalar
    h = toolbox.sig(z) #Apply logistic function
    g = (1 / N) * (np.dot((h - y), x))



    # END TODO
    ###########

    return g
