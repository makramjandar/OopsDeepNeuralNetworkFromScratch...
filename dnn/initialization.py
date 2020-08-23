#!/usr/bin/python
# -*- coding: utf-8 -*-

# @uthor: Makram Jandar 
#   ____ ___   ____ ______ ____   ___ _     ____ _____   ___ ______ ____  _   ___
#  |    |    \|    |      |    |/    | |   |    |     |/    |      |    /   \|    \ 
#   |  ||  _  ||  ||      ||  ||  o  | |    |  ||__/  |  o  |      ||  |     |  _  |
#   |  ||  |  ||  ||_|  |_||  ||     | |___ |  ||   __|     |_|  |_||  |  O  |  |  |
#   |  ||  |  ||  |  |  |  |  ||  _  |     ||  ||  /  |  _  | |  |  |  |     |  |  |
#   |  ||  |  ||  |  |  |  |  ||  |  |     ||  ||     |  |  | |  |  |  |     |  |  |
#  |____|__|__|____| |__| |____|__|__|_____|____|_____|__|__| |__| |____\___/|__|__|
#                                                   © Initialization Functions Class

from numpy import (sqrt, zeros)
from numpy.random import (normal, uniform, seed, randn)

class Initialization():
    ''' Several Initialization Functions '''
    parameters = None

    def __init__(self, layerDims, initType='Default', seed=1):
        """
        Implement the initialization type computation.
        Arguments:
              name -- initType: He(Normal & Uniform) Xavier(Normal & Uniform).
         layerDims -- [L] containing Dims of each layer in our Net.
        Returns:
        parameters -- dict containing parameters "W1", "b1", ..., "WL", "bL":
                Wl -- weight [M] of shape (layerDims[l], layerDims[layer-1]).
                bl -- bias [V] of shape (layerDims[l], 1).
        """

        self.initType = initType
        self.layerDims = layerDims
        self.seed = seed

    ''' Computing initialization '''
    def compute(self):

        parameters = {}
        seed(self.seed)

        for l in range(1, len(self.layerDims)):

            Wl = f'W{l}'
            bl = f'b{l}'

            # Default Heuristic Initialization
            if self.initType == 'Default':
                parameters[Wl] = randn(self.layerDims[l],
                                       self.layerDims[l - 1]) / sqrt(
                                           self.layerDims[l - 1])
                parameters[bl] = zeros((self.layerDims[l], 1))

            # Xavier and He (He-et-al) Initialization
            elif self.initType == 'HeNormal':
                parameters[Wl] = normal(0, sqrt(2.0 / self.layerDims[l - 1]),
                                        (self.layerDims[l], self.layerDims[l - 1]))
                parameters[bl] = normal(0, sqrt(2.0 / self.layerDims[l - 1]),
                                        (self.layerDims[l], 1))

            elif self.initType == 'HeUniform':
                parameters[Wl] = uniform(-sqrt(6.0 / self.layerDims[l - 1]),
                                         sqrt(6.0 / self.layerDims[l - 1]),
                                         (self.layerDims[l], self.layerDims[l - 1]))
                parameters[bl] = uniform(-sqrt(6.0 / self.layerDims[l - 1]),
                                         sqrt(6.0 / self.layerDims[l - 1]),
                                         (self.layerDims[l], 1))

            elif self.initType == 'XavierNormal':
                parameters[Wl] = normal(
                    0, 2.0 / (self.layerDims[l] + self.layerDims[l - 1]),
                    (self.layerDims[l], self.layerDims[l - 1]))
                parameters[bl] = normal(
                    0, 2.0 / (self.layerDims[l] + self.layerDims[l - 1]),
                    (self.layerDims[l], 1))

            elif self.initType == 'XavierUniform':
                parameters[Wl] = uniform(
                    -(sqrt(6.0 / (self.layerDims[l] + self.layerDims[l - 1]))),
                    (sqrt(6.0 / (self.layerDims[l] + self.layerDims[l - 1]))),
                    (self.layerDims[l], self.layerDims[l - 1]))
                parameters[bl] = uniform(
                    -(sqrt(6.0 / (self.layerDims[l] + self.layerDims[l - 1]))),
                    (sqrt(6.0 / (self.layerDims[l] + self.layerDims[l - 1]))),
                    (self.layerDims[l], 1))

        return parameters
