#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:35:53 2020

@author: kalyan
"""

'''## OLS ESTIMATORS (using numpy array)'''

import numpy as np
import matplotlib.pyplot as plt
import statistics as stat # inbuilt function necessary to get mean of a list
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

#Define Function '''Sampling Distribution that recreates a sampling distribution'''

def mpl_for_plotly():
    np.random.seed(1000)
    x = np.random.random(100) ### toy data
    y = np.random.random(100) ### toy data 
    
    ## matplotlib fig
    fig, axes = plt.subplots(2,1, figsize = (10,6))
    axes[0].plot(x, label = 'x')
    axes[1].scatter(x,y)
    
    ## convert and plot in plotly
    plotly_fig = tls.mpl_to_plotly(fig) ## convert 
    iplot(plotly_fig)
    return plotly_fig
