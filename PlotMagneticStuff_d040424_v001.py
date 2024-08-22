#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:53:57 2024

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



df = pd.read_csv('Magnetic.csv')


potentialarr = df['Potential slice'].to_numpy()
Bxarr = df['Bx slice'].to_numpy()
Byarr = df['By slice'].to_numpy()


A = np.reshape(potentialarr, (50, 50))
Bx = np.reshape(Bxarr, (50, 50))
By = np.reshape(Byarr, (50, 50))


fig, ax = plt.subplots(1,1)
pos = ax.imshow(A, cmap='gnuplot', interpolation='none', 
                 vmin=np.min(A), vmax=np.max(A), extent=[0,50,0,50])

fig.colorbar(pos, ax=ax)
ax.set_title("Potential Colour Map Using imshow")
plt.show()

norm = np.sqrt(Bx**2 + By**2)

U=Bx/norm
V=By/norm
fig2, ax2 = plt.subplots(1,1)
q = ax2.quiver(U, V)

plt.show()


def f(r, m, c):
    return m*r + c

df1 = pd.read_csv('distancedataBfield.csv')


distancepot = df1['distance'].to_numpy()
pot = df1['Potential'].to_numpy()
distanceB = df1['distanceB'].to_numpy()
B = df1['B'].to_numpy()

def getindices(array1, array2):
    indices=[]
    for i in range(len(array1)):
        
        if np.isnan(array1[i]) or np.isnan(array2[i])== True:
        
     
            indices.append(i)
         
        if np.isinf(array1[i]) or np.isinf(array2[i])== True:
            
            indices.append(i)
            
        

    return indices

index = getindices(np.log(distancepot), pot)


x = np.delete(np.log(distancepot), index)
y = np.delete(pot, index)


grad, intercept = np.polyfit(np.sort(x)[:50],((np.sort(y))[::-1])[:50], deg=1)


xs=np.sort(np.log(distancepot))

ys = f(xs, grad, intercept)

fig, ax = plt.subplots(1,1)
plt.scatter(np.log(distancepot), pot, s=1, c='k', marker='s')
plt.plot(xs, ys)
plt.title("y="+str(grad)+"x "+str(intercept))
plt.xlabel('log(r)')
plt.ylabel('A')
plt.show()

index1 = getindices(np.log(distanceB), np.log(B))


x1 = np.delete(np.log(distanceB), index1)
y1 = np.delete(np.log(B), index1)


grad1, intercept1 = np.polyfit(np.sort(x1)[:50],((np.sort(y1))[::-1])[:50], deg=1)


xs1=np.sort(np.log(distanceB))

ys1 = f(xs, grad1, intercept1)
print(distanceB)
fig, ax = plt.subplots(1,1)
plt.scatter(np.log(distanceB), np.log(B), s=1, c='k', marker='s')
plt.plot(xs1, ys1)
plt.title("y="+str(grad1)+"x "+str(intercept1))
plt.xlabel('log(r)')
plt.ylabel('log(B)')
plt.show()

