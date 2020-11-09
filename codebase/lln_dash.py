## LAW OF LARGE NUMBERS

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import plotly.express as px
import pandas as pd


def lln_dash(ssize,dist):
    #print(ssize,dist)
    #print(type(ssize),type(dist))
    
    np.random.seed(2000) # set seed for reproducability of results
    
    "PLOT SAMPLE DISTRIBUTION"
    ss = ssize
    dist_type = dist
    
    def normal():
        tmp = np.random.normal(size=ss)
        return tmp 
    def uniform():
        tmp = np.random.uniform(size=ss) #default: a=lowerbound =0.0 ; b=upperbound =1.0
        return tmp 
    def binomial():
        tmp = np.random.binomial(n=10,p=0.5,size=ss) # Num trials = 10, prob= 0.5
        return tmp 
    def poisson():
        tmp = np.random.poisson(lam=2,size=ss) #lam = average occurence of discrete event
        return tmp 
    def logistic():
        tmp = np.random.logistic(loc=0,scale=2,size=ss) #default: loc =mean of the peak= 0; scale =standard deviation=1
        return tmp 
    def multinomial():
        tmp = np.random.multinomial(n=6,pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6],size=ss) #default(dice roll): n= numoutcomes=6; pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
        return tmp 
    def exponential():
        tmp = np.random.exponential(scale=1,size=ss) #default: scale =inverse of rate =1.0.
        return tmp 
    def chisquare():
        tmp = np.random.chisquare(df=2,size=ss) #df - (degree of freedom).
        return tmp 
    def rayleigh():
        tmp = np.random.rayleigh(scale=1.0,size=ss) #default:scale - (standard deviation)=1.0
        return tmp 
    def pareto():
        tmp = np.random.pareto(a=2,size=ss) # a - shape parameter.
        return tmp 
    def zipf():
        tmp = np.random.zipf(a=2, size=ss) # a - shape parameter.
        return tmp
    def the_count():
        print("No distribution chosen")
    
    # Dispatcher aides in calling different functions based on parameters.    
    dispatcher = {
        'Normal': normal, 'Uniform': uniform, 'Binomial': binomial, 'Poisson': poisson, 'Logistic': logistic, 
        'Multinomial': multinomial, 'Exponential': exponential, 'Chi-square': chisquare, 'Rayleigh': rayleigh, 
        'Pareto': pareto, 'Zipf': zipf, 'The_count': the_count
    }

    x = dispatcher[dist_type]() # Assign return value of different function calls (normal,uniform,etc.) to x.
    #print(x)
    
    mu = x.mean()
    sigma = x.std()
    boxtext = '\n'.join((
    			r'Sample Size:%.f ' % (ss, ) ,
    			r'$\mu=%.3f$' % (mu, ),
    			r'$\sigma=%.3f$' % (sigma, )))

    df = pd.DataFrame(x)
    fig = px.histogram(df, x=x, nbins=50, title ="Density of {} distribution - {} r'$\mu$' ".format(dist_type, boxtext), 
			histnorm ='probability', marginal="box", labels=False)
    fig.layout.update( title={ 'font': {'family': "Courier New, monospace", 'size':14, 'color': "RebeccaPurple"}})

    fig.layout.update(shapes=[
		{'type': 'line',
                   'xref': 'x',
                   'yref': 'paper',
                   'x0': mu,
                   'y0': 0,
                   'x1': mu,
                   'y1': 1}])
	
    fig.update_layout(
     annotations=[
        dict(
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Value of a random variable(x)",
            xref="paper",
            yref="paper"
        ),
        dict(
            x=-0.07,
            y=0.5,
            showarrow=False,
            text="Density of (x)",
            textangle=-90,
            xref="paper",
            yref="paper"
        )
    	],)

    return fig
