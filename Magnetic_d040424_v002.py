#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:05:39 2024

@author: panosantonopoulos
"""

import matplotlib
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
import pandas as pd

lx = int(input("Cubic array size: "))
endcondition = float(input("Accuracy: ")) 

nstep = 100000

ly = lx
lz = ly


init_A = np.zeros((ly, lx, lz))
J = np.zeros((ly, lx, lz))

J[:, int(lx/2), int(ly/2)] = 1


def updateA(oldA, J, lx, ly):
    
    
    """
    iup = np.roll(oldA, (0,1,0), axis=(0, -1,0))
    idown = np.roll(oldA, (0,-1,0), axis=(0, -1,0))

    jup = np.roll(oldA, (0,1,0), axis=(0, 1,0))
    jdown = np.roll(oldA, (0,-1,0), axis=(0, 1,0))
    
    #jup = np.roll(oldphi, (1,0,0), axis=(1, 0,0))
    #jdown = np.roll(oldphi, (-1,0,0), axis=(1, 0,0))

    kup = np.roll(oldA, (0,0,1), axis=(1, 0,0))
    kdown = np.roll(oldA, (0,0,-1), axis=(1, 0,0))
    
    #kup = np.roll(oldphi, (1,0,0), axis=(0, -1,0))
    #kdown = np.roll(oldphi, (-1,0,0), axis=(0, -1,0))
    
    """
    iup = np.roll(oldA, 1, axis=2)
    idown = np.roll(oldA, -1, axis=2)
    
    jup = np.roll(oldA, 1, axis=1)
    jdown = np.roll(oldA, -1, axis=1)
    
    kup = np.roll(oldA, 1, axis=0)
    kdown = np.roll(oldA, -1, axis=0)
    
    
    
    newA = (1/6) * ( iup + idown + jdown + jup + kup + kdown + J)
    
    newA[:, 0, :] = 0
    newA[:, :, 0] = 0
    newA[:, (ly-1), :] = 0
    newA[:, :, (lx-1)] = 0
    
    
    
    difference = abs(oldA-newA)
    total = np.sum(difference)
    return newA, total


def calcBfield(A):
    """
    iup = np.roll(A, (0,1,0), axis=(0, -1,0))
    idown = np.roll(A, (0,-1,0), axis=(0, -1,0))

    jup = np.roll(A, (0,1,0), axis=(0, 1,0))
    jdown = np.roll(A, (0,-1,0), axis=(0, 1,0))
    

    
    kup = np.roll(A, (0,0,1), axis=(1, 0,0))
    kdown = np.roll(A, (0,0,-1), axis=(1, 0,0))
    """
    iup = np.roll(A, 1, axis=2)
    idown = np.roll(A, -1, axis=2)
    
    jup = np.roll(A, 1, axis=1)
    jdown = np.roll(A, -1, axis=1)
    
    kup = np.roll(A, 1, axis=0)
    kdown = np.roll(A, -1, axis=0)
    
    
    
    Bx = (jdown-jup)/2
    By = -(idown-iup)/2
    
    return Bx, By


    
previousA = init_A

run = True

while run == True:
    
    updatedA, difference = updateA(previousA, J, lx, ly)
    
    if difference <= endcondition:
        
        run = False
    
    
    
    previousA = updatedA
    
    
Bx, By = calcBfield(updatedA)   
 

norm = np.sqrt(Bx**2 + By**2)
halfslice=int(lx/2)

distancepot = []
pot = []
for i in range(len(updatedA[0])):
    for j in range(len(updatedA[1])):
        
        d =  np.sqrt( (i-25)**2 + (j-25)**2 )
        
        distancepot.append(d)
        
        pot.append(updatedA[halfslice,i,j])


distanceB = []
B = []
for i in range(len(norm[0])):
    for j in range(len(norm[1])):
        
        d =  np.sqrt( (i-25)**2 + (j-25)**2 )
        
        distanceB.append(d)
        
        B.append(norm[halfslice,i,j])



data = {'Potential slice': updatedA[halfslice,:,:].flatten(), 'Bx slice': Bx[halfslice,:,:].flatten(), 
        'By slice': By[halfslice,:,:].flatten()} 

col = ['Potential slice', 'Bx slice', 'By slice'] # Pandas dataframe parameters 

df = pd.DataFrame(data, columns = col) # Create pandas dataframe

df.to_csv('Magnetic.csv', index=False) # Save data 
data1 = {'distance':distancepot, 'Potential': pot, 'distanceB':distanceB, 'B':B} 

col1 = ['distance', 'Potential', 'distanceB', 'B'] # Pandas dataframe parameters 

df1 = pd.DataFrame(data1, columns = col1) # Create pandas dataframe

df1.to_csv('distancedataBfield.csv', index=False) # Save data 


