import matplotlib.pyplot as plt
import scipy.stats
# from colour import Color
import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
# go.renderers.default = "vscode"
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)


def generate_config():
    return {'showLink': False, 'displayModeBar': False, 'showAxisRangeEntryBoxes': True}


def plot_high_low(prices, lookback_high, lookback_low, ticker, title):
    config = generate_config()
    layout = go.Layout(title=title)

    stock_trace = go.Scatter(
        name=ticker,
        x=prices.index,
        y=prices,
        line={'color': 'red'})
    high_trace = go.Scatter(
        x=lookback_high.index,
        y=lookback_high,
        name='Column lookback_high',
        fill = None,
        line={'color':'#2D3ECF'})
    low_trace = go.Scatter(
        x=lookback_low.index,
        y=lookback_low,
        name='Column lookback_low',
        fill='tonexty',
        fillcolor='rgba(0,250,0,0.4)',
        line={'color': '#B6B2CF'})

    pyo.iplot({'data': [stock_trace, high_trace, low_trace], 'layout': layout}, config=config)

# def plot_bolinger(prices,title):
#     fig = go.Figure()
#     layout = go.Layout(title=title)

#     fig.add_traces(go.Scatter(
#         name='close',
#         x=prices.index,
#         y=prices.close,
#         line={'color': 'black'}))
#     fig.add_traces(go.Scatter(
#         name='sma_band',
#         x=prices.index,
#         y=prices.sma,
#         line={'color': 'red'}))
#     fig.add_traces(go.Scatter(
#         x=prices.index,
#         y=prices.upper,
#         name='high_band',
#         fill=None,
#         line={'color':'#2D3ECF'}))
#     fig.add_traces(go.Scatter(
#         x=prices.index,
#         y=prices.lower,
#         name='low_band',
#         fill='tonexty',
#         fillcolor='rgba(0,250,0,0.4)',
#         line={'color': '#B6B2CF'}))

#     pyo.iplot({'data': fig, 'layout': layout})

def plot_bar(data,factor):
    """
    Plots bucketed returns
    """
    
    plot = data[[factor,'target-1','target-2','target-3','target-4','target-5']].groupby(factor).mean().plot.bar()
    plt.title('Factor of'+ ' '+ factor)    
    
    return plot