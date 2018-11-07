#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Brettmccausland
"""
import math
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator
from operator import itemgetter, attrgetter
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

#precondiotion:
#
#
#postcondition: value at x is returned
def Guassian_Partial_Pivot(matrix_A):
  sum=0
  count=len(matrix_A)
  print(count)
  # for i in range(count-1):
    # loop


def GPP_forward(matrix_A,vec_rowmax):
  sum=0
  count=len(poly_coef)
  for i in range(count-1):
    sum = (sum+poly_coef[i])*x
  return sum + poly_coef[count-1]

#precondiotion:
#
#postcondition:
def Guassian_Elimination(poly_coef,x):
  sum=0
  count=len(poly_coef)
  for i in range(count-1):
    sum = (sum+poly_coef[i])*x
  return sum + poly_coef[count-1]



#----- Question 1 Write a code to solve it by Gaussian elimination  ------
#             with scaled partial pivoting. Carry out the calculation
#             with four decimal places.
 #EQ1 = 0.4096 0.1234 0.3678 0.2943 0.4043
 #EQ2 = 0.2246 0.3872 0.4015 0.1129 0.1550
 #EQ3 = 0.3645 0.1920 0.3781 0.0643 0.4240
 #EQ4 = 0.1784 0.4002 0.2786 0.3927 0.2557
#  matrix_A= [ 0.4096, 0.1234, 0.3678, 0.2943, 0.4043],
# [ 0.2246, 0.3872, 0.4015, 0.1129, 0.1550],
# [ 0.3645, 0.1920, 0.3781, 0.0643, 0.4240],
# [0.1784 ,0.4002 ,0.2786 ,0.3927 ,0.2557]

data=pd.read_csv('matrixA.csv')  #import the data set
# Guassian_Partial_Pivot(matrix_A)
#----------- Question 2 ----------
matrix_B=[0.3840, 0.5124, 0.7890, 1.2718, 0.5432, 0.8774, 0.9125],
[-0.1127, 0.0358, 0.4230, 0.2879, 0.3750, 0.1248],
[2.3715, 0.7887, -4.5612, 3.6233, 0.7819, -2.1352, 0.1435]

# (d1, d2, d3, d4, d5, d6, d7) = (0.3840, 0.5124, 0.7890, 1.2718, 0.5432, 0.8774, 0.9125)
# (a1, a2, a3, a5, a6, a7) = (−0.1127, 0.0358, 0.4230, 0.2879, 0.3750, 0.1248)
# b1, b2, b3, b4, b5, b6, b7) = (2.3715, 0.7887, −4.5612, 3.6233, 0.7819, −2.1352, 0.1435)
