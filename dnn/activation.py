#!/usr/bin/python
# -*- coding: utf-8 -*-

# @uthor: Makram Jandar
#   /    |  /  ]      |    |  |  |/    |      |    /   \|    \ 
#  |  o  | /  /|      ||  ||  |  |  o  |      ||  |     |  _  |
#  |     |/  / |_|  |_||  ||  |  |     |_|  |_||  |  O  |  |  |
#  |  _  /   \_  |  |  |  ||  :  |  _  | |  |  |  |     |  |  |
#  |  |  \     | |  |  |  | \   /|  |  | |  |  |  |     |  |  |
#  |__|__|\____| |__| |____| \_/ |__|__| |__| |____\___/|__|__|
#                  © Activation and derivatives Functions Class

from numpy import (exp, array, maximum, square, log, tanh, arctan, ones_like,
                   where, clip)

""" Several Activation and derivatives Functions """
class Activation():
    def __init__(self, function, derivative, Z, dA):
        self.function = function
        self.derivative = derivative
        self.Z = Z
        self.dA = dA

    ''' Computing activations and derivatives '''
    def compute(self):
        
        # Sigmoid function and derivative
        if self.function == 'Sigmoid':
            dZ = 1 / (1 + exp(-self.Z))
            return self.dA * (dZ * (1 - dZ)) if self.derivative else dZ
        
        # ReLU function and derivative
        elif self.function == 'ReLU':
            if self.derivative:
                dZ = array(self.dA, copy=True)
                dZ[self.Z <= 0] = 0
                return dZ
            return maximum(0, self.Z)
        
        # TanH function and derivative
        elif self.function == 'TanH':
            return 1 - square(self.Z) if self.derivative else tanh(self.Z)
        
        # ArcTan function and derivative
        elif self.function == 'ArcTan':
            return 1 / (1 + square(self.Z)) if self.derivative else arctan(self.Z)
        
        # Identity function and derivative
        elif self.function == 'Identity':
            return ones_like(self.Z) if self.derivative else self.Z
        
        # ELU function and derivative
        elif self.function == 'ELU':
            if self.derivative:
                return where(self.Z < 0, alpha * (exp(self.Z)), 1)
            return where(self.Z < 0, alpha * (exp(self.Z) - 1), self.Z)
        
        # LeakyReLU function and derivative
        elif self.function == 'LeakyReLU':
            if self.derivative:
                return where(self.Z > 0, self.Z, self.Z * 0.01)
            return clip(self.Z > 0, 0.01, 1.0)
        
        # SoftPlus function and derivative
        elif self.function == 'SoftPlus':
            if self.derivative:
                return 1 / (1 + exp(-array(self.Z)))
            return log(1 + exp(self.Z))