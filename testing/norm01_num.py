#!/usr/bin/python

import numpy as np


A = np.array([[ 1, 2, 3, 4] , [1, 2, 3, 4]])

print(A)

nA = np.linalg.norm(A, axis=0)

print(nA)
