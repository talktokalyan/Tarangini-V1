'''## CENTRAL LIMIT THEOREM '''
### For use by Dash Application - Tarangini
### Uses plotly graph objects - histogram trace- to output two figures.
### Figure 1 is a density / probability distribution of a random sample.
### Figure 2 is a sampling distribution of the underlying random variables.


import numpy as np
import matplotlib.pyplot as plt
import statistics as stat # inbuilt function necessary to get mean of a list
from   codebase.llnbase import lln
import plotly.graph_objs as go

#Define Function '''Sampling Distribution that recreates a sampling distribution of means'''

def clt_sampling_dist(ss,dist_type,rep):
    # Housekeeping
    plt.close()
    np.random.seed(1500)
    R = rep # Number of Replications of the sampling
    dist = dist_type
    N = ss # Size of Sample
    
    mean_array = np.empty(R) # Array that holds means of each Replication sample. Initialized to empty.
    
    for i in range(R):
            tmp = lln(N,dist) # Call 'lln' from LLN.py & assign return value to array tmp.
            mean_tmp = tmp.mean() # Mean of one sample
            #print(i, mean_tmp)
            mean_array[i] = mean_tmp
    #print(mean_array)

	#y, R, N = clt_sampling_dist(R,N,'Uniform') # Function call & Assignment: Sampling Means Array 
    
    mom = stat.mean(mean_array) # Mean of Sample Means
    stdom = stat.stdev(mean_array) # Analytical Standard error of Sampling Distribution
    boxtext = '\n'.join((
            r'Num. Replications=%i' % (R, ),
            r'Sample-Size=%i' % (N, ),
            r'$\mu=%3.3f$' % (mom, ),
            r'Std Err=%.3f' % (stdom, ) 
            ))
    props = dict(boxstyle='round', facecolor='cyan', alpha=0.5)
    
    #one sample distribution
    fig1 = go.Figure(go.Histogram(x=tmp, histnorm='probability', nbinsx=50, marker={'color': "grey"}))
   
    #distribution of sample means - Sampling distribution
    fig2 = go.Figure(go.Histogram(x=mean_array, nbinsx=50, marker = {'color': "green"}))

 

    fig1.layout.update(
		    title_text='Distribution of values in one random sample', # title of plot
    		    xaxis_title_text='Value of random variable (x)', # xaxis label
    		    yaxis_title_text='Density', # yaxis label
		shapes=[
		{'type': 'line',
                   'xref': 'x',
                   'yref': 'paper',
                   'x0': mean_tmp,
                   'y0': 0,
                   'x1': mean_tmp,
                   'y1': 1,
		   'line': {'color': "Black", 'width': 4, 'dash': "dash"},}])

    fig2.layout.update(
		    title_text='Sampling Distribution of Mean values', # title of plot
    		    xaxis_title_text='Mean value of each sample', # xaxis label
    		    yaxis_title_text='Density', # yaxis label
		shapes=[
		{'type': 'line',
                   'xref': 'x',
                   'yref': 'paper',
                   'x0': mom,
                   'y0': 0,
                   'x1': mom,
                   'y1': 1,
		   'line': {'color': "Black", 'width': 4, 'dash': "dashdot"},
		   }])

    return fig1,fig2