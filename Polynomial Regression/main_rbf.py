#!/usr/bin/env python

import json

import matplotlib.pyplot as plt
import numpy as np

import rbf
from plot_rbf import plot_rbf

"""
Computational Intelligence TU - Graz
Assignment 1: Linear Regression
Part 2: With radial basis non-linear features

This file:
1) loads the data from 'data.json'
2) trains and test a linear regression model for a given number of RBF centers
3) plots the results

TODO boxes are to be found in 'rbf.py'
"""
__author__ = 'bellec,subramoney'


def main():
    # Set the n_centers of the polynomial expansion
    n_centers = 5

    data_path = 'data.json'

    # Load the data
    f = open(data_path, 'r')
    data = json.load(f)
    for k, v in data.items():
        data[k] = np.array(v).reshape((len(v), 1))

    # Print the training and testing errors
    theta, err_train, err_val, err_test = rbf.train_and_test(data, n_centers)
    print('Training error {} \t Validation error {} \t Testing error {} '.format(err_train, err_val, err_test))

    plot_rbf(data, n_centers, theta)
    plt.savefig('RBF' + str(n_centers) + '.png')
    plt.show()


if __name__ == '__main__':
    main()
