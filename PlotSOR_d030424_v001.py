#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:57:01 2024

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



df = pd.read_csv('sor.csv')


w = df['omega'].to_numpy()
iterations = df['iterations'].to_numpy()

fig, ax = plt.subplots(1,1)
plt.plot(w, iterations)
plt.xlabel("Omega")
plt.ylabel("Iterations")
plt.show()

        
        

















