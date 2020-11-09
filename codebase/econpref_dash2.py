#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:52:05 2020

@author: kalyan
"""

## COMMON UTILITY FUNCTIONS - NON CRS Cobb Douglas

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import plotly.graph_objects as go

#plt.style.use('seaborn-white')

def iso_utility(prf,xmax=20,ymax=20,xshare=0.5,yshare=0.5):
    preference = prf
    x = np.linspace(0,xmax,50)
    y = np.linspace(0,ymax,50)
    X,Y = np.meshgrid(x,y)
    a = xshare
    b = yshare
    #np.random.seed(200) # set seed for reproducability of results

    
    def cobbdouglas():
        tmp = (X**a)*(Y**b)
        return tmp
    def leontief():
        c = np.vstack([x, y]) # concatenate column-wise
        tmp = np.nanmin(c, axis = 0) #find minimum value in each row.
        return tmp 
    def perfectsubstitute():
        tmp = (a*X) + (b*Y) #default: a=lowerbound =0.0 ; b=upperbound =1.0
        return tmp 

    # Dispatcher aides in calling different functions based on parameters.    
    dispatcher = {
        'Cobb-Douglas': cobbdouglas, 'Perfect-Complements': leontief, 'Perfect-Substitutes': perfectsubstitute
    }

    U = dispatcher[preference]() # Assign return value of different function calls (normal,uniform,etc.) to x.
    print(U)
    
    fig1 = go.Figure(data=[go.Surface(z = U)])

    fig1.update_layout(title='3D Utility Function ({})'.format(preference), autosize=False,
                  width=500, height=500,scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                  margin=dict(l=65, r=50, b=65, t=90))
    
    fig1.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

    fig2 = go.Figure(data = go.Contour(z = U, 
                                      contours = dict(coloring='lines',showlabels = True), 
                                      line = dict(width=2, color= 'black')),
                    layout = go.Layout(height=500, width=500, 
                                       title ='Iso-Utility Curves (Indifference curves)-({})'.format(preference))
                   )

    #plt.contour(X, Y, Z, 10, colors='blue')  
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #plt.title("ISO-UTILITY CURVES-({})".format(preference))
    #plt.legend
    #plt.savefig('utility.png', bbox_inches="tight")
    return fig1, fig2

iso_utility('Cobb-Douglas',xmax=20,ymax=20,xshare=0.5,yshare=0.5)