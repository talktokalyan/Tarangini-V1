"""
Author: Kalyan Kolukuluri
Start date: July 7th 2020

This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location. There are two callbacks,
one uses the current location to render the appropriate page content, the other
uses the current location to toggle the "active" properties of the navigation
links.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plot.ly/urls
"""
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from codebase.lln_dash import *
from codebase.clt_dash import *
from codebase.classic_ols_dash import *
from codebase.mpl_example import *
from codebase.market_dash import *
from codebase.econpref_dash import *
from codebase.econcost_dash import *


app = dash.Dash(external_stylesheets=[dbc.themes.MINTY])

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "black",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "20rem",
    "margin-right": "2rem",
    "padding": "5rem 1rem",
    "color" : "white",
}

sidebar = html.Div(
    [
        html.H2("Tarangini", className="display-4"),
        html.Hr(),
        html.P("Simulation & visualization for Economics & Econometrics", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Law of Large Numbers", href="/page-1", id="page-1-link"),
                dbc.NavLink("Central Limit Theorem", href="/page-2", id="page-2-link"),
                dbc.NavLink("OLS Estimators", href="/page-3", id="page-3-link"),
                dbc.NavLink("Market Statics", href="/page-4", id="page-4-link"),
                dbc.NavLink("Consumer Preferences", href="/page-5", id="page-5-link"),
                dbc.NavLink("Econ Cost functions", href="/page-6", id="page-6-link"),

            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

# App Layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# Example figure
colors = {'background': '#111111','text': '#7FDBFF'}
df = pd.DataFrame({"x": [1, 2, 3], "SF": [4, 1, 2], "Montreal": [2, 4, 5]})
fig = px.bar(df, x="x", y=["SF", "Montreal"], barmode="group")
fig.update_layout(plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'])

# List of Distributions -P1, P2
dist_list = ['Normal', 'Uniform', 'Binomial', 'Poisson', 'Logistic', 'Multinomial', 'Exponential', 'Chi-square', 'Rayleigh', 'Pareto', 'Zipf' ]

# List of Distributions - P5, P6
pref_list = ['Cobb-Douglas', 'Perfect-Substitutes' ]

#-----------------------
# Page 1 Controls
controls_page1 = html.Div(
    [	#Heading Row
        dbc.Row(dbc.Col([html.H1("1-Law of Large Numbers"), html.Hr()])),
	
	#Controls Row
        dbc.Row([dbc.Col([html.Label('Underlying Distribution Type',style={'size':20}), dcc.Dropdown(id = 'dist_type_p1', options= [{'label': i, 'value': i} for i in dist_list],)], width=2),
		dbc.Col(html.P('Placeholder for future controls'), width=2),
                dbc.Col([html.Label('Sample Size'), dcc.Slider(id="sample_size_p1", min=10, max=10000, step= 100, value=1000, marks={'100':100,'1000':1000,'5000':5000,'10000':10000})]),
            ]),
	dbc.Row(dbc.Col(html.Hr())),

	#Output Row
	dbc.Row(dbc.Col(html.Div(id='output_container_p1'))),
    ])

#-----------------------
# Page 2 Controls
controls_page2 = html.Div(
    [	#Heading Row
        dbc.Row(dbc.Col([html.H1("2-Central Limit Theorem"), html.Hr()])),
	
	#Controls Row
        dbc.Row([dbc.Col([html.Label('Underlying Distribution',style={'size':20}), dcc.Dropdown(id = 'dist_type_p2', options= [{'label': i, 'value': i} for i in dist_list],)], width=2),
		dbc.Col(html.P('Placeholder for future controls'), width=2),
        dbc.Col([html.Label('Sample Size'), dcc.Slider(id="sample_size_p2", min=10, max=1000, step= 10, value=500, marks={'10':10,'100':100,'500':500,'1000':1000}),],width=3),
        dbc.Col([html.Label('Num of Replications'), dcc.Slider(id="num_rep_p2", min=10, max=1000, step= 10, value=500, marks={'10':10,'100':100,'500':500,'1000':1000}),],width=3),
        dbc.Col(html.Button('Run Simulation', id='run_sim_button_p2', n_clicks=0, style={'BackgroundColor':'white'}),width=2) 
    ]),
	dbc.Row(dbc.Col(html.Hr())),

	#Output Row
	#dbc.Row(dbc.Col(html.Div(id='output_container_p2'))),
	html.Div(id='output_container_p2'),
    ])

#-----------------------
# Page 3 Controls
controls_page3 = html.Div(
    [	#Heading Row
        dbc.Row(dbc.Col([html.H1("3- Unbiased Estimators - OLS"), html.Hr()])),
	
	#Controls Row
        dbc.Row([
        dbc.Col(html.P('Placeholder for future controls'), width=2),
		dbc.Col([html.Label('Population Parameters(Beta)',style={'size':20},), dcc.Input(id = 'beta_pop_list_p3',type= "text",disabled=True)], width=2),
        dbc.Col([html.Label('Sample Size'), dcc.Slider(id="sample_size_p3", min=10, max=1000, step= 10, value=500, marks={'10':10,'100':100,'500':500,'1000':1000}),],width=3),
        dbc.Col([html.Label('Num of Replications'), dcc.Slider(id="num_rep_p3", min=10, max=1000, step= 10, value=500, marks={'10':10,'100':100,'500':500,'1000':1000}),],width=3),
        dbc.Col(html.Button('Run Simulation', id='run_sim_button_p3', n_clicks=0, style={'BackgroundColor':'white'}),width=2) 
        ]),
        dbc.Row(dbc.Col(html.Hr())),

	#Output Row
	#dbc.Row(dbc.Col(html.Div(id='output_container_p3'))),
	html.Div(id='output_container_p3'),
    ])

#-------------------
# Page 4 Controls
controls_page4 = html.Div(
    [	#Heading Row
        dbc.Row(dbc.Col([html.H1("4- Linear Demand and Supply. Market Statics"), html.Hr()])),
	
	#Controls Row
        dbc.Row([
        dbc.Col(html.P(['Demand Curve Spec: Q = a - b(Price)',html.Br(),'Supply Curve Spec: Q = c - d(Price)']), width=3),
		dbc.Col([html.Label('Demand Intercept (a):',style={'size':20},), dcc.Slider(id="a_p4", min=0, max=20, step= 5, value=2, marks={i: '{}'.format(i) for i in range(0,20,5)}),],width=2),
		dbc.Col([html.Label('Demand Slope (b):',style={'size':20},), dcc.Slider(id="b_p4", min=0, max=20, step= 1, value=2,  marks={i: '{}'.format(i)  for i in range(0,20,5)}),],width=2),
		dbc.Col([html.Label('Supply Intercept (c):',style={'size':20},), dcc.Slider(id="c_p4", min=-5, max=20, step= 1, value=-2,  marks={i: '{}'.format(i)  for i in range(-5,20,5)}),],width=2),
		dbc.Col([html.Label('Supply Slope (d):',style={'size':20},), dcc.Slider(id="d_p4", min=0, max=20, step= 1, value=3,  marks={i: '{}'.format(i)  for i in range(0,20,5)}),],width=2),
        dbc.Col([html.Label('Lump sum tax (t):',style={'size':20},), dcc.Slider(id="tax_p4", min=0, max=5, step= 1, value=2,  marks={i: '{}'.format(i) for i in range(0,10,5)}),],width=2),
        ]),
        dbc.Row([dbc.Col(html.Button('Run Simulation', id='run_sim_button_p4', n_clicks=0, style={'BackgroundColor':'white'}),width=2)]), 
        dbc.Row(dbc.Col(html.Hr())),

	#Output Row
	html.Div(id='output_container_p4'),
    ])

#-------------------
# Page 5 Controls
controls_page5 = html.Div(
    [	#Heading Row
        dbc.Row(dbc.Col([html.H1("5- Economic Preferences and Utility functions"), html.Hr()])),
	
        #Controls Row
        dbc.Row([dbc.Col([html.Label('Underlying Preferences',style={'size':20}), dcc.Dropdown(id = 'pref_type_p5', options= [{'label': i, 'value': i} for i in pref_list],)], width=3),
		dbc.Col([html.Label('Max X:',style={'size':20},), dcc.Slider(id="xmax_p5", min=2, max=20, step= 5, value=10, marks={i: '{}'.format(i) for i in range(0,20,5)}),],width=2),
		dbc.Col([html.Label('Max Y:',style={'size':20},), dcc.Slider(id="ymax_p5", min=2, max=20, step= 5, value=10,  marks={i: '{}'.format(i)  for i in range(0,20,5)}),],width=2),
		dbc.Col([html.Label('Share of Good X (a):',style={'size':20},), dcc.Input(id="xshare_p5", type="number", min=0, max=1, step=0.05, value=0.5),],width=3),
                ]),
        dbc.Row([dbc.Col(html.Button('Run Simulation', id='run_sim_button_p5', n_clicks=0, style={'BackgroundColor':'white'}),width=2)]), 
        dbc.Row(dbc.Col(html.Hr())),

	#Output Row
	#dbc.Row(dbc.Col(html.Div(id='output_container_p3'))),
	html.Div(id='output_container_p5'),
])
#-------------------
# Page 6 Controls
controls_page6 = html.Div(
    [	#Heading Row
        dbc.Row(dbc.Col([html.H1("6- Economic Cost functions"), html.Hr()])),
	
        #Controls Row
        dbc.Row([dbc.Col([html.Label('Enter Variable Cost function exactly in format (e.g. 3*x**2-2*x)',style={'size':20}), dcc.Input(id = 'var_cost_p6', type="text" )], width=5),
		dbc.Col([html.Label('Fixed Cost of Manufacturing:',style={'size':20},), dcc.Input(id="fix_cost_p6",type="number", min=0, step= 20, value=100,),],width=2),
        dbc.Col([html.Label('Max X:',style={'size':20},), dcc.Input(id="xmax_p6",type="number", min=5, max=100, step= 5, value=10,),],width=2),
                ]),
        dbc.Row([dbc.Col(html.Button('Run Simulation', id='run_sim_button_p6', n_clicks=0, style={'BackgroundColor':'white'}),width=2)]), 
        dbc.Row(dbc.Col(html.Hr())),

	#Output Row
	#dbc.Row(dbc.Col(html.Div(id='output_container_p3'))),
	html.Div(id='output_container_p6'),
])
#-------------------
# CALLBACKS
#-------------------
# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on

@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 10)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 10)]

#-------------------

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", "/page-1"]:
        return controls_page1
    elif pathname == "/page-2":
        return controls_page2
    elif pathname == "/page-3":
        return controls_page3
    elif pathname == "/page-4":
        return controls_page4
    elif pathname == "/page-5":
        return controls_page5
    elif pathname == "/page-6":
        return controls_page6
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ])

#-------------------

# Page 1 Callback
@app.callback(
    Output(component_id='output_container_p1', component_property='children'),
    [Input(component_id="dist_type_p1", component_property='value'), Input(component_id='sample_size_p1', component_property='value')]
)
def update_output_page1(dist_type_value,sample_size_value):
    fig_p1 = lln_dash(sample_size_value, dist_type_value)
    return dcc.Graph(id='lln_fig_p1', figure=fig_p1)

#-------------------

# Page 2 Callback
@app.callback(
    Output(component_id='output_container_p2', component_property='children'),
    [Input(component_id='run_sim_button_p2', component_property='n_clicks')],
    [State(component_id="dist_type_p2", component_property='value'), State(component_id='sample_size_p2', component_property='value'), State(component_id='num_rep_p2', component_property='value')]
)
def update_output_page2(n_clicks,dist_type_value,sample_size_value,num_replications_value):
    fig_1_p2, fig_2_p2 = clt_sampling_dist(sample_size_value, dist_type_value,num_replications_value )
    sim_info_p2 = 'This simulation has drawn {} random samples of {} distribution each; sample size = {}.'.format(num_replications_value,dist_type_value,sample_size_value)
    return html.Div([
                    dbc.Row(dbc.Col (html.P(sim_info_p2))), 
                    dbc.Row([dbc.Col(dcc.Graph(id='cln_fig1_p2', figure= fig_1_p2)), dbc.Col(dcc.Graph(id='cln_fig2_p2', figure= fig_2_p2))])
                    ])

#-------------------

# Page 3 Callback
@app.callback(
    Output(component_id='output_container_p3', component_property='children'),
    [Input(component_id='run_sim_button_p3', component_property='n_clicks')],
    [State(component_id='sample_size_p3', component_property='value'), State(component_id='num_rep_p3', component_property='value')]
)
def update_output_page3(n_clicks,sample_size_value,num_replications_value):
    
    mylist=[1,1.5]
    fig_1_p3 = get_2dscatter(sample_size_value, mylist )    
    fig_2_p3, beta_hat_val = get_sampling_betahat(sample_size_value,num_replications_value, mylist)
    sim_info_p3 = 'This simulation has drawn {} random samples of X,Y Variables. Each sample size has {} cases/records. <br> Further Simple OLS is run on each sample and Beta estimates are plotted (Right graph).'.format(num_replications_value,sample_size_value)
    return html.Div([
                    dbc.Row(dbc.Col (html.P(sim_info_p3))), 
                    dbc.Row([dbc.Col(dcc.Graph(id='b_hat_fig1_p3', figure= fig_1_p3)), dbc.Col(dcc.Graph(id='b_hat_fig2_p3', figure= fig_2_p3))])
                    ])
#-------------------

# Page 4 Callback
@app.callback(
    Output(component_id='output_container_p4', component_property='children'),
    [Input(component_id='run_sim_button_p4', component_property='n_clicks')],
    [State(component_id='a_p4', component_property='value'), State(component_id='b_p4', component_property='value'),
     State(component_id='c_p4', component_property='value'), State(component_id='d_p4', component_property='value'),
     State(component_id='tax_p4', component_property='value'),]
)

def update_output_page4(n_clicks,a,b,c,d,tax):
    #fig_1_p3 = get_2dscatter(sample_size_value, mylist )   
    if a < c:
        sim_info_p4 = "Input value of a must be greater than c. Else it will be as if demand is always lower than supply."
    fig_1_p4 = get_market_plot(a,b,c,d,tax)
    #fig_2_p4 = mpl_for_plotly()
    sim_info_p4 = 'Input Market Parameters: a = {}, b = {}, c = {}, d = {} & tax = {}.'.format(a,b,c,d,tax)
    return html.Div([
                    dbc.Row([dbc.Col(dcc.Graph(id='market_fig1_p4', figure= fig_1_p4)), dbc.Col(html.P(sim_info_p4))])
                    ])
#-------------------

# Page 5 Callback
@app.callback(
    Output(component_id='output_container_p5', component_property='children'),
    [Input(component_id='run_sim_button_p5', component_property='n_clicks')],
    [State(component_id='pref_type_p5', component_property='value'),
     State(component_id='xmax_p5', component_property='value'), State(component_id='ymax_p5', component_property='value'),
     State(component_id='xshare_p5', component_property='value'),]
    )
def update_output_page4(n_clicks,pref,xmax,ymax,xshare):
    #fig_1_p3 = get_2dscatter(sample_size_value, mylist )   
    if xshare > 1:
        sim_info_p5 = "You have chosen a Non Constant returns function. a + b not equal 1."
    fig_1_p5, fig_2_p5 = iso_utility(pref,xmax,ymax,xshare)
    #fig_2_p5 = mpl_for_plotly()
    sim_info_p5 = 'Input Preference Parameters: Preference Type = {}, Max X = {}, Max Y = {}, a = {} , b = {}.'.format(pref,xmax,ymax,xshare,1-xshare)
    return html.Div([
                    dbc.Row(dbc.Col (html.P(sim_info_p5))), 
                    dbc.Row([dbc.Col(dcc.Graph(id='pref_fig1_p5', figure= fig_1_p5)), dbc.Col(dcc.Graph(id='pref_fig2_p5', figure= fig_2_p5))])
                    ])
#-------------------

# Page 6 Callback
@app.callback(
    Output(component_id='output_container_p6', component_property='children'),
    [Input(component_id='run_sim_button_p6', component_property='n_clicks')],
    [State(component_id='var_cost_p6', component_property='value'),
     State(component_id='fix_cost_p6', component_property='value'),
     State(component_id='xmax_p6', component_property='value'),]
    )
def update_output_page4(n_clicks,var_cost_value,fix_cost_value, xmax_value):
    fig_1_p6 = get_cost_functions(var_cost_value,fix_cost_value,xmax_value)
    sim_info_p6 = 'Input Parameters: Cost Function: Variable cost = {}, Fixed cost = {}, Max Output (x) = {}.'.format(var_cost_value,fix_cost_value,xmax_value)
    return html.Div([
                    dbc.Row(dbc.Col (html.P(sim_info_p6))), 
                    dbc.Row([dbc.Col(dcc.Graph(id='cost_fig1_p6', figure= fig_1_p6)), dbc.Col(html.P(sim_info_p6))])
                    ])
#-------------------


if __name__ == "__main__":
    app.run_server(port=8050, debug=False)
