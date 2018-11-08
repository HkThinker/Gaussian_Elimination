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
def Guassian_Partial_Pivot(df_A,list_rowmax):
  
  numRows=df_A.shape[0] 
  numcol=df_A.shape[1]
  pivots=[]
  # get max rows
  list_rowmax = Row_abs_max(df_A)
  # forward elimination
  GPP_forward(df_A,list_rowmax)
  
  # back subsitution
  # for i in range(count-1):
    # loop

def Row_abs_max(df_A):
  numRows=df_A.shape[0] 
  numcol=df_A.shape[1]
  list_r=[]
  for r in range(numRows):
      maxv=0
      for c in range(numcol-1):
          value=df_A.iloc[r,c] #
          print('value:',value)
          value=abs(value)
          if(value>maxv):
              maxv=value
              print('maxv:',maxv)
      list_r.append(maxv)
  print('list_r:',list_r)
  return list_r

def partialPivotRow(df_A,list_rowmax,row,const_col):
  numRows=df_A.shape[0] 
  numcol=df_A.shape[1] 
  c=const_col 
  value= df_A.iloc[r,c]
  maxv = abs(value/list_rowmax[r])
  pivot_index = row
  for r in range(row,numRows):
    value=df_A.iloc[r,c]
    value=abs(value/list_rowmax[r])
    if(value>maxv):
        pivot_index=r
        maxv=value  
  return pivot_index
    
def GPP_forward(df_A,list_rowmax):
 
 numRows=df_A.shape[0] 
 numcol=df_A.shape[1]
 

#precondiotion:
#
#postcondition:
#def Guassian_Elimination(poly_coef,x):




#----- Question 1 Write a code to solve it by Gaussian elimination  ------
#             with scaled partial pivoting. Carry out the calculation
#             with four decimal places.
df_A=pd.read_csv('matrixA.csv')  #import the data set
list_rowmax=[]
list_rowmax = Row_abs_max(df_A)
Guassian_Partial_Pivot(df_A,list_rowmax)
v= partialPivot(df_A,list_rowmax,1,0)
print('v',v)













# Guassian_Partial_Pivot(matrix_A)
#----------- Question 2 ----------
df_B=[0.3840, 0.5124, 0.7890, 1.2718, 0.5432, 0.8774, 0.9125],
[-0.1127, 0.0358, 0.4230, 0.2879, 0.3750, 0.1248],
[2.3715, 0.7887, -4.5612, 3.6233, 0.7819, -2.1352, 0.1435]

# (d1, d2, d3, d4, d5, d6, d7) = (0.3840, 0.5124, 0.7890, 1.2718, 0.5432, 0.8774, 0.9125)
# (a1, a2, a3, a5, a6, a7) = (−0.1127, 0.0358, 0.4230, 0.2879, 0.3750, 0.1248)
# b1, b2, b3, b4, b5, b6, b7) = (2.3715, 0.7887, −4.5612, 3.6233, 0.7819, −2.1352, 0.1435)
