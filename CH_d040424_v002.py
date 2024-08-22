#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:21:28 2024

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

phi0 = float(input("Phi0: "))
dx = float(input("Delta x: ")) 
dt = float(input("Delta t: ")) 
nstep = 100000
lx = 100
ly = lx
a = 0.1
M = 0.1
b = a
k = 0.1

init_phi = np.random.uniform(low=-0.1, high=.1, size=(ly,lx)) + phi0

fig = plt.figure()
im = plt.imshow(init_phi, animated=True)
plt.colorbar()
plt.show()

init_mu = np.zeros((ly, lx), dtype=float)


def update(a, b, k, dx, dt, oldphi):
    """
    oldphi_iup = np.roll(oldphi, (1,0), axis=(1, 0))
    oldphi_idown = np.roll(oldphi, (-1,0), axis=(1, 0))
    oldphi_jup = np.roll(oldphi, (-1,0), axis=(0, 1))
    oldphi_jdown = np.roll(oldphi, (1,0), axis=(0, 1))
    """
    
    oldphi_jup = np.roll(oldphi, 1, axis=0) #jup
    oldphi_jdown = np.roll(oldphi, -1, axis=0) #jdown
    oldphi_idown = np.roll(oldphi, -1, axis=1) #idown
    oldphi_iup = np.roll(oldphi, 1, axis=1) #iup
    
    
    newmu = -a*oldphi + b*oldphi**3 - (k/dx**2) * (   
            oldphi_iup + oldphi_idown + oldphi_jup + oldphi_jdown - 4*oldphi )
    """
    mu_iup = np.roll(newmu, (1,0), axis=(1, 0))
    mu_idown = np.roll(newmu, (-1,0), axis=(1, 0))
    mu_jup = np.roll(newmu, (-1,0), axis=(0, 1))
    mu_jdown = np.roll(newmu, (1,0), axis=(0, 1))
    """
    
    mu_jup = np.roll(newmu, 1, axis=0) #jup
    mu_jdown = np.roll(newmu, -1, axis=0) #jdown
    mu_idown = np.roll(newmu, -1, axis=1) #idown
    mu_iup = np.roll(newmu, 1, axis=1) #iup
    
    
    newphi = oldphi + (M*dt/dx**2) * ( mu_iup + mu_idown +
                                      mu_jup + mu_jdown - 4*newmu )
    
    
    return newphi

def freeEnergy(a, k, dx, phi):
    """
    phi_iup = np.roll(phi, (1,0), axis=(1, 0))
    phi_idown = np.roll(phi, (-1,0), axis=(1, 0))
    
    phi_jup = np.roll(phi, (-1,0), axis=(0, 1))
    phi_jdown = np.roll(phi, (1,0), axis=(0, 1))
    """
    phi_jup = np.roll(phi, 1, axis=0) #jup
    phi_jdown = np.roll(phi, -1, axis=0) #jdown
    phi_idown = np.roll(phi, -1, axis=1) #idown
    phi_iup = np.roll(phi, 1, axis=1) #iup
    
    
    gradphisq = ( (phi_jup-phi_jdown)/(2*dx) )**2 + ( (phi_iup-phi_idown)/(2*dx) )**2
    
    #gradphisq = ( (phi_jup-phi_jdown)/(2*dx) +  (phi_iup-phi_idown)/(2*dx) )**2
    f = -(a/2)*phi**2 + (a/4)*phi**4 + (k/2)*gradphisq
    
    return np.sum(f)
    
    

previousphi = init_phi
previousmu = init_mu          

freeenergy = []
sweep = []

for n in range(nstep):
    
    newphi = update(a, b, k, dx, dt, previousphi)
    
    if (n%10)==0:
        sweep.append(n)
        freeenergy.append(freeEnergy(a, k, dx, newphi))
        
    previousphi = newphi
    
    if (n%50)==0:
        plt.cla()
        im = plt.imshow(newphi, animated=True, vmin=-1, vmax=1) 
        plt.draw()
        plt.pause(0.000001)
        


data = {'Sweep': sweep, 'Free Energy Density': freeenergy} 

col = ['Sweep', 'Free Energy Density'] # Pandas dataframe parameters 

df = pd.DataFrame(data, columns = col) # Create pandas dataframe

df.to_csv('FreeEnergyData.csv', index=False) # Save data 



