#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:47:12 2024

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



df = pd.read_csv('FreeEnergyData.csv')

sweep = df['Sweep'].tolist()

freeenergy = df['Free Energy Density'].tolist()

fig1 = plt.figure()
plt.scatter(sweep, freeenergy,  s=1, c='k', marker='s')
plt.xlabel("Sweep")
plt.ylabel("Free Energy")

plt.show()