#!/usr/bin/env python
__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Logistic Regression

This file contains generic implementation of gradient descent solvers
The functions are:
- TODO gradient_descent: for a given function with its gradient it finds the minimum with gradient descent
- TODO adaptative_gradient_descent: Same with adaptative learning rate
"""

import numpy as np


def gradient_descent(f, df, x0, learning_rate, max_iter):
    '''
    Find the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the current parameter vector x by the gradient times learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: gradient of f
    :param x0: initial point
    :param learning_rate: l
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (array of errors over iterations)
    '''
    ##############
    #
    # TODO
    #
    # Implement a gradient descent algoritm

    E_list = np.zeros(max_iter)
    x = x0
    for i in range(0, max_iter):
        x = x - learning_rate * df(x)
        E_list[i] = f(x)
    # END TODO
    ###########

    return x, E_list


def adaptative_gradient_descent(f, df, x0, initial_learning_rate, max_iter):
    '''
    Find the optimal solution of the function f using an adaptative gradient descent:

    After every update check whether the cost increased or decreased.
        - If the cost increased, reject the update (go back to the
        previous parameter setting) and multiply the learning rate by 0.7.
        - If the cost decreased, accept the
        update and multiply the learning rate by 1.03.

    The iteration count should be increased after every iteration even if the update was rejected.

    :param f: function to minimize
    :param df: gradient of f
    :param x0: initial point
    :param learning_rate: initial learning rate l0
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (list of errors), l_rate (The learning rate at the final iteration)
    '''

        ##############
    #
    # TODO
    #
    # Implement a gradient descent algorithm
    #

    l_rate = initial_learning_rate
    E_list = np.zeros(max_iter)
    x = x0
    E_list[0] = f(x0)
    for i in range(1, max_iter):
        lastx = x
        x = x - l_rate * df(x)
        E_list[i] = f(x)

        if E_list[i] > E_list[i-1]:
            l_rate *= 0.7
            x = lastx
        else:
            l_rate *= 1.03
    print(l_rate)
    return x, E_list, l_rate

    # END TODO
    ###########


