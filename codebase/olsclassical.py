'''## OLS ESTIMATORS (using numpy array)'''
# Classical variant of OLS, where all classical assumptions are satisfied.
# Function '''Sampling Distribution that recreates a sampling distribution'''

import numpy as np
import matplotlib.pyplot as plt
import statistics as stat # inbuilt function necessary to get mean of a list
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#%matplotlib inline
'''-----------------------------------------------------------------------------'''
''' GET BETA_HAT ESTIMATES FOR ONE SAMPLE '''
'''-----------------------------------------------------------------------------'''
def get_betahat(ss, beta_pop_input = [1,1]):
    
    #np.random.seed(2000) ## Do not put seed so that we get varied results.

    ## Assign function values to local var.
    N = ss # Size of Sample
    K = len(beta_pop_input) # Number of parameters to be estimated.
    beta_pop = np.array([beta_pop_input]).transpose() # Convert row elements of population parameters into a column (Kx1) dimension
    print('beta_pop', type(beta_pop), beta_pop, beta_pop.shape)
    
    ## ONE SAMPLE CODE
    ## create matrix of explnatory variables (X) and explained variables (Y)
    X = np.hstack([np.ones((N,1)), np.reshape(np.array([np.random.normal(0,1,(K-1)*N)]),(N, K-1))]) # necessary to create (NxK) 2-D array in Python 
    #print('X', type(X), X, X.shape)
    err_pop = np.reshape(np.array([np.random.normal(0,1,N)]),(N,1)) # necessary to create (Nx1) 2-D array in Python
    Y = X.dot(beta_pop) + err_pop
    
    ## classical beta_hat estimates
    beta_hat = np.linalg.inv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(Y))
    print('beta_hat', type(beta_hat), beta_hat, beta_hat.shape)
    
    #fig1 = px.scatter_3d(df, x = X[:,0], y=X[:,1], z= Y )

    # plt.scatter(X[:,1],Y)
    # plt.scatter(X[:,0],Y)
    # plt.show()
    return beta_hat
'''-----------------------------------------------------------------------------'''
''' GET SAMPLING BETA_HAT ESTIMATES FOR A MONTE_CARLO SIMULATION'''
'''-----------------------------------------------------------------------------'''

def get_sampling_betahat(ss, rep, beta_pop_input = [1,1]):
    
    ## Assign function values to local var.
    N = ss # Size of Sample
    R = rep # Num of Monte Carlo simulations / replications.
    K = len(beta_pop_input) # Number of parameters to be estimated.
    beta_pop = beta_pop_input # Convert row elements of population parameters into a column (Kx1) dimension
    
    beta_hat_array = np.empty((R,K),dtype=float) # container to be populated with calculated values.
   
    # Run Montecarlo Simulation
    for i in range(R):
       tmp = get_betahat(N,beta_pop).transpose() # call get_betahat function 
       beta_hat_array[i,:] = tmp # beta_hat_array stores beta estimates in columns.
    
    df = pd.DataFrame(beta_hat_array)
    
    beta_hat_mean = np.mean(beta_hat_array[:,1]) # Mean of beta_hat Means (x1 parameter)
    beta_hat_std = np.std(beta_hat_array[:,1]) # Analytical Standard error of Sampling Distribution

    boxtext = "BOXTEXT"
    fig2 = px.histogram(df, x=beta_hat_array[:,1], nbins=50, marginal="violin", labels=False)

    fig2.layout.update(
		title = "Sampling Distribution of Beta hat {}".format( boxtext), # title of plot
        xaxis = {'title' : {'text':'Value of Beta Hat (beta1_hat)','font': {'family': "Courier New, monospace", 'size':12, 'color': "Magenta"}}}, # xaxis label
        yaxis = {'title' : {'text':'Count of Beta Hat (beta1_hat)','font': {'family': "Courier New, monospace", 'size':12, 'color': "Magenta"}}}, # yaxis label
		shapes=[
		{'type': 'line',
                 'xref': 'x',
                 'yref': 'paper',
                 'x0': beta_hat_mean,
                 'y0': 0,
                 'x1': beta_hat_mean,
                 'y1': 1,
                 'line': {'color': "Black", 'width': 2, 'dash': "dash"},},
        {'type': 'line',
                 'xref': 'x',
                 'yref': 'paper',
                 'x0': beta_pop_input[1],
                 'y0': 0,
                 'x1': beta_pop_input[1],
                 'y1': 1,
                 'line': {'color': "Red", 'width': 2.5},},]
        )
   
    fig2.layout.update( title={ 'font': {'family': "Courier New, monospace", 'size':14, 'color': "RebeccaPurple"}})
    
    return fig2, beta_hat_array


'''-----------------------------------------------------------------------------'''
''' GET 2D scatter of a sample  '''
'''-----------------------------------------------------------------------------'''
def get_2dscatter(ss, beta_pop_input = [1,1]):
    
    #np.random.seed(2000) ## Do not put seed so that we get varied results.

    ## Assign function values to local var.
    N = ss # Size of Sample
    K = len(beta_pop_input) # Number of parameters to be estimated.
    beta_pop = np.array([beta_pop_input]).transpose() # Convert row elements of population parameters into a column (Kx1) dimension
    
    ## ONE SAMPLE CODE
    ## create matrix of explnatory variables (X) and explained variables (Y)
    X = np.hstack([np.ones((N,1)), np.reshape(np.array([np.random.normal(0,1,(K-1)*N)]),(N, K-1))]) # necessary to create (NxK) 2-D array in Python 
    #print('X', type(X), X, X.shape)
    err_pop = np.reshape(np.array([np.random.normal(0,1,N)]),(N,1)) # necessary to create (Nx1) 2-D array in Python
    Y = X.dot(beta_pop) + err_pop
    
    df_x =pd.DataFrame(X)
    df_y =pd.DataFrame(Y)
    df = pd.concat([df_x,df_y], axis=1)
    df.columns=['intercept','x1','y']
    print(df)
    '''fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')    
    ax.scatter(X[:,0], X[:,1], Y, c='g', marker='o')
    ax.set_xlabel('x1 values')
    ax.set_ylabel('x2 values')
    ax.set_zlabel('Y values')
    fig1.show()
    '''
    ## classical beta_hat estimates
    
    fig1 = px.scatter(df, x = 'x1', y = 'y')
    return fig1
#mylist=[1,1]
#fig = get_2dscatter(100,mylist)
