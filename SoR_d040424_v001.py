#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:52:37 2024

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
import time
lx = int(input("Cubic array size: "))
endcondition = float(input("Accuracy: ")) 
updatecondition = str(input("Jacobi (j), Gauss-Sidel (g) or SoR (s) update rule: "))
nstep = 100000

ly = lx
lz = ly


init_phi = np.zeros((ly, lx, lz))
rho = np.zeros((ly, lx, lz))

rho[int(ly/2), int(lx/2), int(lz/2)] = 1


def Jacobi(oldphi, rho, lx, ly):

    iup = np.roll(oldphi, (0,1,0), axis=(0, -1,0))
    idown = np.roll(oldphi, (0,-1,0), axis=(0, -1,0))

    jup = np.roll(oldphi, (0,1,0), axis=(0, 1,0))
    jdown = np.roll(oldphi, (0,-1,0), axis=(0, 1,0))
    
    #jup = np.roll(oldphi, (1,0,0), axis=(1, 0,0))
    #jdown = np.roll(oldphi, (-1,0,0), axis=(1, 0,0))

    kup = np.roll(oldphi, (0,0,1), axis=(1, 0,0))
    kdown = np.roll(oldphi, (0,0,-1), axis=(1, 0,0))
    
    #kup = np.roll(oldphi, (1,0,0), axis=(0, -1,0))
    #kdown = np.roll(oldphi, (-1,0,0), axis=(0, -1,0))
    
    """
    iup = np.roll(oldphi, 1, axis=2)
    idown = np.roll(oldphi, -1, axis=2)
    
    jup = np.roll(oldphi, 1, axis=1)
    jdown = np.roll(oldphi, -1, axis=1)
    
    kup = np.roll(oldphi, 1, axis=0)
    kdown = np.roll(oldphi, -1, axis=0)
    """
    newphi = (1/6) * ( iup + idown + jdown + jup + kup + kdown + rho)
    
    newphi[:, 0, :] = 0
    newphi[:, :, 0] = 0
    newphi[:, (ly-1), :] = 0
    newphi[:, :, (lx-1)] = 0
    newphi[0,:,:] = 0
    newphi[(lz-1),:,:]=0
    
    
    difference = abs(oldphi-newphi)
    total = np.sum(difference)
    return newphi, total

    

def GS(phi, rho, lx, ly, lz):
    
    oldphi = np.copy(phi)
    
    for k in range(lz-1):
        
        for j in range(lx-1):
            
            for i in range(ly-1):
                
                if k ==(lx-1) or k ==0 or j ==(lx-1) or j==0 or i ==(lx-1) or i==0:
                
                    phi[k,j,i]=0
                
                else:
                    
                
                    phi[k,j,i] = (1/6)*(phi[k,j+1,i]+phi[k,j-1,i]
                                           +phi[k,j,i+1]+phi[k,j,i-1]
                                           +phi[k+1,j,i]+phi[k-1,j,i]
                                           +rho[k,j,i])

    difference = abs(oldphi-phi)
    total = np.sum(difference)
    return phi, total

def SoR(phi, w, rho, lx, ly, lz):
    
    oldphi = np.copy(phi)
    
    for k in range(lz-1):
        
        for j in range(lx-1):
            
            for i in range(ly-1):
                
                if k ==(lx-1) or k ==0 or j ==(lx-1) or j==0 or i ==(lx-1) or i==0:
                
                    phi[k,j,i]=0
                
                else:
                    
                
                    phi[k,j,i] = w*(1/6)*(phi[k,j+1,i]+phi[k,j-1,i]
                                           +phi[k,j,i+1]+phi[k,j,i-1]
                                           +phi[k+1,j,i]+phi[k-1,j,i]
                                           +rho[k,j,i]) + (1-w)*phi[k,j,i]

    difference = abs(oldphi-phi)
    total = np.sum(difference)
    return phi, total


def calcEfield(phi):
    
    iup = np.roll(phi, (0,1,0), axis=(0, -1,0))
    idown = np.roll(phi, (0,-1,0), axis=(0, -1,0))

    jup = np.roll(phi, (0,1,0), axis=(0, 1,0))
    jdown = np.roll(phi, (0,-1,0), axis=(0, 1,0))
    
    kup = np.roll(phi, (0,0,1), axis=(1, 0,0))
    kdown = np.roll(phi, (0,0,-1), axis=(1, 0,0))
    """
    iup = np.roll(phi, 1, axis=2)
    idown = np.roll(phi, -1, axis=2)
    
    jup = np.roll(phi, 1, axis=1)
    jdown = np.roll(phi, -1, axis=1)
    
    kup = np.roll(phi, 1, axis=0)
    kdown = np.roll(phi, -1, axis=0)
    
    """
    Ex = -(idown-iup)/2 
    Ey = -(jdown-jup)/2
    Ez = -(kdown-kup)/2
    
    return Ex, Ey, Ez

    


w = np.arange(1,1.7, 0.01)
iterations = []
for i in range(len(w)):
    start=time.time()
    init_phi = np.zeros((ly, lx, lz))
    rho = np.zeros((ly, lx, lz))

    rho[int(ly/2), int(lx/2), int(lz/2)] = 1
    previousphi = init_phi

    run = True
    n=0
    #print(w[i])
    while run == True:
        
        if updatecondition == 'j':
            updatedphi, difference = Jacobi(previousphi, rho, lx, ly)
        
        elif updatecondition == 'g':
            updatedphi, difference = GS(previousphi, rho, lx, ly,lz)
            
        elif updatecondition == 's':
            updatedphi, difference = SoR(previousphi, w[i], rho, lx, ly,lz)
            n=n+1
        
        #difference = abs(oldsum-newsum)
        
        if difference <= endcondition:
            
            run = False
        
        previousphi = updatedphi 
    iterations.append(n)
    print("Just did Omega= ",w[i], " which took ", time.time()-start, " seconds with ", n, " iterations")

Ex, Ey, Ez = calcEfield(updatedphi)   
 
print(iterations, len(iterations))
print(w, len(w))
norm = np.sqrt(Ex**2 + Ey**2 + Ez**2)
slicepoint = int(lx/2)

data = {'omega': w, 'iterations': iterations} 

col = ['omega', 'iterations'] # Pandas dataframe parameters 

df = pd.DataFrame(data, columns = col) # Create pandas dataframe

df.to_csv('sor.csv', index=False) # Save data 

