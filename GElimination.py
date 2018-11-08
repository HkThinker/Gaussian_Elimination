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
  list_rowmax.clear
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
#precondition:
  #list_rowmax= list of abs values with indexs maping to df_a rows
  #
#postcondition:
  
def partialPivotRow(df_A,list_rowmax,row,c):
  numRows=df_A.shape[0] 
 
  value= df_A.iloc[row,c].copy
  maxv = abs(value/list_rowmax[row])
  pivot_index = row
  for r in range(row,numRows):
    value=df_A.iloc[r,c]
    value=abs(value/list_rowmax[r])
    if(value>maxv):
        pivot_index=r, maxv=value
  
  return pivot_index
    
def GPP_forward(df_a,list_rowmax):
 # row of pivot, col of pivot =r_p,c_p
 numRows=df_A.shape[0]-1
 numcol=df_A.shape[1]
 for r_p in range(numRows): #for every diagnol
   c_p=r_p
   pivotrow=partialPivotRow(df_A,list_rowmax,r_p,c_p) #find pivot row
   if(pivotrow!=r_p):                     #if pivot row is not in diagnol swap
     DF_and_list_RowSwap(df_a,r_p,pivotrow,list_rowmax)
   for r in range(r_p+1, numRows-1): #for every row after the pivot row
     f =df_A.iloc[r,c_p].copy/df_A.iloc[r_p,c_p].copy; # get elimination factor of col 
     for c  in range(c_p,numcol-1): #apply elimination factor to every col in row starting at col pivot
         pivotvalue = df_A.iloc[r_p,c].copy * f
         df_A.iloc[r,c]-=pivotvalue


def DF_RowSwap(df_a,r1,r2):
  temp = df_a.iloc[r1].copy()
  df_a.iloc[r1] = df_a.iloc[r2]
  df_a.iloc[r2] = temp
def DF_and_list_RowSwap(df_a,r1,r2,l_a):
 DF_RowSwap(df_a,r1,r2)
 l_a[r1], l_a[r2] = l_a[r2], l_a[r1]
 
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















# Guassian_Partial_Pivot(matrix_A)
#----------- Question 2 ----------
df_B=[0.3840, 0.5124, 0.7890, 1.2718, 0.5432, 0.8774, 0.9125],
[-0.1127, 0.0358, 0.4230, 0.2879, 0.3750, 0.1248],
[2.3715, 0.7887, -4.5612, 3.6233, 0.7819, -2.1352, 0.1435]

# (d1, d2, d3, d4, d5, d6, d7) = (0.3840, 0.5124, 0.7890, 1.2718, 0.5432, 0.8774, 0.9125)
# (a1, a2, a3, a5, a6, a7) = (−0.1127, 0.0358, 0.4230, 0.2879, 0.3750, 0.1248)
# b1, b2, b3, b4, b5, b6, b7) = (2.3715, 0.7887, −4.5612, 3.6233, 0.7819, −2.1352, 0.1435)
