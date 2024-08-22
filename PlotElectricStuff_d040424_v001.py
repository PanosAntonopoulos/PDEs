#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:35:42 2024

@author: panosantonopoulos
"""

import matplotlib

#matplotlib.use('TKAgg')

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator
import pandas as pd
import time
from scipy.optimize import curve_fit


df = pd.read_csv('Poisson.csv')


potentialarr = df['Potential slice'].to_numpy()
Exarr = df['Ex slice'].to_numpy()
Eyarr = df['Ey slice'].to_numpy()
Ezarr = df['Ez slice'].to_numpy()

phi = np.reshape(potentialarr, (50, 50))
Ex = np.reshape(Exarr, (50, 50))
Ey = np.reshape(Eyarr, (50, 50))
Ez = np.reshape(Ezarr, (50, 50))
norm = np.sqrt(Ex**2 + Ey**2 + Ez**2)

fig, ax = plt.subplots(1,1)
pos = ax.imshow(phi, cmap='gnuplot', interpolation='none', 
                 vmin=np.min(phi), vmax=np.max(phi), extent=[0,50,0,50])

fig.colorbar(pos, ax=ax)
ax.set_title("Potential Colour Map Using imshow")
plt.show()



U=Ex/norm
V=Ey/norm
fig2, ax2 = plt.subplots(1,1)
q = ax2.quiver(U, V)

plt.show()

def f(r, m, c):
    return m*r + c


df1 = pd.read_csv('distancedata.csv')


distancepot = df1['distance'].to_numpy()
pot = df1['Potential'].to_numpy()
distanceE = df1['distanceE'].to_numpy()
E = df1['E'].to_numpy()


def getindices(array1, array2):
    indices=[]
    for i in range(len(array1)):
        
        if np.isnan(array1[i]) or np.isnan(array2[i])== True:
        
     
            indices.append(i)
         
        if np.isinf(array1[i]) or np.isinf(array2[i])== True:
            
            indices.append(i)
            
        

    return indices


index = getindices(np.log(distancepot), np.log(pot))


x = np.delete(np.log(distancepot), index)
y = np.delete(np.log(pot), index)


grad, intercept = np.polyfit(np.sort(x)[:50],((np.sort(y))[::-1])[:50], deg=1)


xs=np.sort(np.log(distancepot))

ys = f(xs, grad, intercept)



fig, ax = plt.subplots(1,1)
plt.scatter(np.log(distancepot), np.log(pot), s=1, c='k', marker='s')
plt.plot(xs, ys)
plt.title("y="+str(grad)+"x "+str(intercept))
plt.xlabel('log(r)')
plt.ylabel('log(phi)')
plt.show()

index1 = getindices(np.log(distanceE), np.log(E))


x1 = np.delete(np.log(distanceE), index1)
y1 = np.delete(np.log(E), index1)


grad1, intercept1 = np.polyfit(np.sort(x1)[:50],((np.sort(y1))[::-1])[:50], deg=1)


xs1=np.sort(np.log(distanceE))

ys1 = f(xs, grad1, intercept1)

fig, ax = plt.subplots(1,1)
plt.scatter(np.log(distanceE), np.log(E), s=1, c='k', marker='s')
plt.plot(xs1, ys1)
plt.title("y="+str(grad1)+"x "+str(intercept1))
plt.xlabel('log(r)')
plt.ylabel('log(E)')
plt.show()







        
        

















