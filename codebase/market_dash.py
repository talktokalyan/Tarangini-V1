#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 17:48:23 2020

@author: kalyan
"""

'''## Create a Generic Supply and Demand Graph'''
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import plotly.tools as tls
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

class Market:

    """ Demand function: Qd = a - b*(price) ; Supply function: Qs = c + d*(price - tax). """
    """ 1. Equilibrium finding methods: price(), quantity(), consumer_surp(), producer_surp(), tax_rev() """ 
    """ 2. Non-equilibrium methods: inv_demand(quantity), inv_supply(quantity), inv_supply_no_tax(quantity)  """

    def __init__(self, a, b, c, d, tax):
        """
        Set up market parameters.  All parameters are scalars.  See
        https://lectures.quantecon.org/py/python_oop.html for interpretation.

        """
        self.a, self.b, self.c, self.d, self.tax = a, b, c, d, tax
        if a < c:
            raise ValueError('Insufficient demand. a must be greater than c')

    def price(self):
        "Return equilibrium price"
        return  (self.a - self.c + self.d * self.tax) / (self.b + self.d)

    def quantity(self):
        "Compute equilibrium quantity"
        return  self.a - self.b * self.price()

    def consumer_surp(self):
        "Compute consumer surplus"
        # == Compute area under inverse demand function == #
        integrand = lambda x: (self.a / self.b) - (1 / self.b) * x
        area_u_demand, error = quad(integrand, 0, self.quantity())
        return area_u_demand - self.price() * self.quantity()

    def producer_surp(self):
        "Compute producer surplus"
        #  == Compute area above inverse supply curve, excluding tax == #
        integrand = lambda x: -(self.c / self.d) + (1 / self.d) * x
        area_u_supply, error = quad(integrand, 0, self.quantity())
        return (self.price() - self.tax) * self.quantity() - area_u_supply

    def tax_rev(self):
        "Compute tax revenue"
        return self.tax * self.quantity()

    def inv_demand(self, x):
        "Compute inverse demand"
        return self.a / self.b - (1 / self.b)* x

    def inv_supply(self, x):
        "Compute inverse supply curve"
        return -(self.c / self.d) + (1 / self.d) * x + self.tax

    def inv_supply_no_tax(self, x):
        "Compute inverse supply curve without tax"
        return -(self.c / self.d) + (1 / self.d) * x
    
'''--------------------------------------------------------'''
 
def get_market_plot(a=3,b=1.5,c=-2,d = 2.5,tax = 0):
        
    baseline_params = a, b, c, d, tax
    m = Market(*baseline_params) # Creates an instance of Market Class.
    print(m.quantity(),m.price())
    q_max = m.quantity() * 3
    p_max = m.price()*3
    q_grid = np.linspace(0.0, q_max, 100)
    pd = m.inv_demand(q_grid)
    ps = m.inv_supply(q_grid)
    psno = m.inv_supply_no_tax(q_grid)

    fig, ax = plt.subplots()
    ax.plot(q_grid, pd, '-',lw=2.5, alpha=0.6, label='Demand: $Q^d$ = {}-{}(P)'.format(a,b))
    ax.plot(q_grid, ps, '-',  lw=2.5, alpha=0.6, label='Supply: $Q^s$ = {}+{}(P-{})'.format(c,d,tax))
    ax.plot(q_grid, psno, '--', lw=2.5, alpha=0.6, label='Supply w/o tax:$Q^s$ = {}-{}(P)'.format(c,d))
    ax.scatter(m.quantity(),m.price(), label='Equilibrium(Q={},P={})'.format(m.quantity(),m.price()))
    ax.vlines(m.quantity(),0,m.price(),linestyle=':',lw=1.5)
    ax.hlines(m.price(),0,m.quantity(),linestyle=':',lw=1.5)

    ax.set_xlabel('Quantity', fontsize=14)
    ax.set_xlim(0,q_max)
    ax.set_ylim(0,p_max)
    ax.set_ylabel('Price', fontsize=14)
    ax.legend(loc='upper center', frameon=False, fontsize=6)
    plt.title('Demand and Supply of a Good (Linear Demand)')
    #plt.fill_between(m.quantity(),)
    #plt.show()
    
    ## convert and plot in plotly
    plotly_fig = tls.mpl_to_plotly(fig) ## convert 
    #iplot(plotly_fig)
    return plotly_fig

'''--------------------------------------------------------'''

# Test
#get_market_plot(10,2,5,3,1)