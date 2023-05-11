#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 08 18:34:04 2022

@author: jheyston
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def compute_sk_yk(sk, yk, mk, gk, lbfgsLayers, iter):

    if iter == 0:
        sk[0, :, :, :] = np.copy(mk)
        yk[0, :, :, :] = np.copy(gk)

    if iter > 0:
        sk[0, :, :, :] = mk - sk[0,:, :, :]
        sk[1:lbfgsLayers - 1, :, :, :] = sk[0:lbfgsLayers - 2, :, :, :]
        sk[0, :, :] = mk

        yk[0, :, :, :] = gk - yk[0, :, :, :]
        yk[1:lbfgsLayers - 1, :, :, :] = yk[0:lbfgsLayers - 2, :, :, :]
        yk[0, :, :, :] = gk

def L_BFGS( sk, yk, gk, lbfgsLayers, iter):
    counter = 0

    sigma = np.zeros((lbfgsLayers))
    epsilon = np.zeros((lbfgsLayers))

    if iter < lbfgsLayers:
        layers = iter
    else:
        layers = lbfgsLayers-1

    ''' Computing gamma '''
    if sum(sum(sum(yk[counter+1, :, :, :]*yk[counter+1, :, :, :]))) == 0:
        gamma = 0
    else:
        gamma = sum(sum(sum(sk[counter+1, :,:, :]*yk[counter+1, :, :, :] )))/sum(sum(sum(yk[counter+1, :, :, :]*yk[counter+1, :, :, :])))

    q = np.copy(gk)

    ''' Computing r '''
    for counter in range(layers):
        # Computing sigma
        if sum(sum(sum(yk[counter + 1, :, :, :] * sk[counter + 1, :, :, :]))) == 0:
            sigma[counter] = 0
        else:
            sigma[counter] = 1.0 / sum(sum(sum(yk[counter + 1, :, :, :] * sk[counter + 1, :, :, :])))

        # Computing epsilon
        epsilon[counter] = sigma[counter] * sum(sum(sum(sk[counter + 1, :, :, :] * q)))
        # Updating q
        q = q - epsilon[counter]*yk[counter+1, :, :, :]

    r = gamma * q

    for counter in range(layers, 0, -1):
        # Computing beta
        beta = sigma[counter-1] * sum(sum(sum(yk[counter, :, :,:] * r)))
        # Updating r
        r = r + sk[counter, :, :, :] * (epsilon[counter - 1] - beta)

    return r
