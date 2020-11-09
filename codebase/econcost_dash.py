#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:12:19 2020

@author: kalyan
"""

from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def get_cost_functions(var_cost = '3*x**2-2*x' , fix_cost=0, max_output=10):
    xmax = max_output
    vc = var_cost
    fc = fix_cost
    tc =  vc + "+" + str(fc)  # sympy module
    x = symbols('x')
    mc = diff(tc,x)

    print(tc, mc)
    vc_x = lambdify(x, vc, modules=['numpy']) # sympy module
    tc_x = lambdify(x, tc, modules=['numpy']) # sympy module
    mc_x = lambdify(x, mc, modules=['numpy']) # sympy module
    #atc_x = lambdify(x, atc, modules=['numpy']) # sympy module

    xvals = np.linspace(1,xmax,50)
    print(xvals)
    trace_avc = go.Scatter(x= xvals, y= vc_x(xvals)/xvals, name='Average Variable Cost (AVC)')
    trace_afc = go.Scatter(x= xvals, y= fc/xvals, name='Average Fixed Cost (AFC)')
    trace_atc = go.Scatter(x= xvals, y= tc_x(xvals)/xvals, name='Average Total Cost (ATC)')
    #trace_tc = go.Scatter(x= xvals, y= tc_x(xvals), name='Total Cost (TC)')
    trace_mc = go.Scatter(x= xvals, y= mc_x(xvals), name='Marginal Cost (MC)')


    cost_layout = go.Layout(
        title = dict(text='Economic Cost functions for ({}) in Output(x) terms'.format(tc)),
        xaxis=dict(title="Number of Units of Output (x)"),
        yaxis=dict(title="Cost components")
        )
    fig = go.Figure(data=[trace_avc, trace_afc, trace_atc, trace_mc], layout = cost_layout)
    #fig.show()
    return fig
