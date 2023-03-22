# --------------------------------------------------------------------------------------------------------
# -------------------- All libraries, variables and functions are defined in this fil --------------------
# --------------------------------------------------------------------------------------------------------


import pandas as pd

import hvplot.pandas
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

from sklearn.cluster import KMeans

import plotly.graph_objects as go

# 1. libraries ------------------------------------------------------------------------------------------/
# a-1) main dependencies and setup
from package.constants import * # constants


# b-1) plotting

# -------------------------------------------------------------------------------------------------------/

# Plotting function --------------------------------------------------------------------------------/
def line (df, chart_title):
    # Create a list of traces for each column in the DataFrame
    traces = []
    for i, col in enumerate(df.columns):
        col_name = col.split("_")
        trace = go.Scatter(x=df.index,
                           y=df[col],
                           name=col_name[-1],
                           mode='lines',
                           line=dict(color=SEVENSET[i%len(SEVENSET)]),
                          )
        traces.append(trace)
    # Create the layout
    layout = go.Layout(title=dict(text=chart_title,
                                  font=dict(size= 24, color= 'black', family= "Times New Roman"),
                                  x=0.5,
                                  y=0.9),
                       width=1000,
                       height=600,
                       legend=dict(
                           yanchor="top",
                           y=0.99,
                           xanchor="left",
                           x=0.01,
                           bgcolor= '#f7f7f7',
                           font=dict(color='black')),
                       xaxis=dict(title='Crypto',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True), 
                       yaxis=dict(title='Price Change (%)',
                                  color= 'black',
                                  showline=True,
                                  linewidth=1,
                                  linecolor='black',
                                  mirror=True),
                       plot_bgcolor='#f7f7f7',
                       paper_bgcolor="#f7f7f7")

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)
    # Show the figure
    fig.show()

def histogram (df, bins, location):
    # Set the figure size
    plt.figure(figsize=(FIG_HEIGHT, FIG_WIDTH))

    #Plot the Clusters
    ax = sns.scatterplot(data = df_market_scaled,
                         x = 'price_change_percentage_24h',
                         y = 'price_change_percentage_7d', 
                         hue = km.labels_, 
                         palette = 'colorblind', 
                         alpha = 0.8, 
                         s = 150,
                         legend = False)

    #Plot the Centroids
    ax = sns.scatterplot(data = cluster_centers, 
                         x = 'price_change_percentage_24h',
                         y = 'price_change_percentage_7d', 
                         hue = cluster_centers.index, 
                         palette = 'colorblind', 
                         s = 600,
                         marker = 'D',
                         ec = 'black', 
                         legend = False)

    # Add Centroid Labels
    for i in range(len(cluster_centers)):
                   plt.text(x = cluster_centers.price_change_percentage_24h[i], 
                            y = cluster_centers.price_change_percentage_7d[i],
                            s = i, 
                            horizontalalignment='center',
                            verticalalignment='center',
                            size = 15,
                            weight = 'bold',
                            color = 'white')
# -------------------------------------------------------------------------------------------------------/
