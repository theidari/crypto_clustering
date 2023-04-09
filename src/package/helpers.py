# --------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------ All libraries, variables and functions are defined in this file ---------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------

# main dependencies and setup
import pandas as pd

# ml dependencies and setup
from sklearn.cluster import KMeans # KMeans
from sklearn.decomposition import PCA # PCA
from sklearn.preprocessing import StandardScaler # StandardScale to resize the distribution of values 
from sklearn.metrics import silhouette_score # Silhouette method
from sklearn.metrics import calinski_harabasz_score # Calinski Harabasz method

# plotting dependencies and setup  
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# package dependencies and setup
from package.constants import * # constants

# --------------------------------------------------------------------------------------------------------------------------------------------
# functions
# clustering function ________________________________________________________________________________________________________________________
def clusters_methods(df, methods):
    methods_list = []
    optimal_ks = []
    for method in methods:
        scores = []
        
        # create a for loop to compute the inertia with each possible value of k
        for k in range(2, 11):
            
            # create a KMeans model using the loop counter for the n_clusters
            km = KMeans(n_clusters=k, n_init=40, random_state=1)
            
            # fit the model to the data using dataframe
            km.fit(df)
            
            # append the model to the inertia list
            # wcss elbow 
            if method == "wcss_elbow":
                scores.append(km.inertia_)
                
            #others    
            else:
                query = f"""scores.append({method}_score(df, km.labels_))"""
                exec(query)
                
        # create a series with the data
        method_series = pd.Series(scores, index=range(2, 11), name=method.replace("_", " ").title())
        
        #finding best k
        if method == "wcss_elbow": # for elbow method
            
            # calculate the percentage of variance explained for each value of k
            ms_index = list(method_series.index)
            pve = [100 * (1 - (method_series[i] / method_series[ms_index[0]])) for i in ms_index]

            # Find the elbow point (i.e., the value of k where the PVE starts to level off)
            threshold = 11
            for i in range(1, len(pve)):
                if abs(pve[i] - pve[i-1]) < threshold:
                    optimal_k = i + 1
                    break
                    
        elif method == "silhouette": # for silhouette method
            point = method_series.max()
            optimal_k = method_series.index[method_series == point][0]

        elif method == "calinski_harabasz": # for calinski method
            optimal_k = method_series.idxmax()
                    
        # create list of results
        methods_list.append(method_series)
        optimal_ks.append(optimal_k)
        
    return methods_list, optimal_ks

# plotting functions
# line chart _________________________________________________________________________________________________________________________________
def line (df, chart_title):
    # create a list of traces for each column in the DataFrame
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
        
    # create the layout
    layout = go.Layout(title=dict(text=chart_title,
                                  font=dict(size= 20, color= 'black', family= "Times New Roman"),
                                  x=0.5,
                                  y=0.9),
                       width=1200,
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

    # create the figure
    fig = go.Figure(data=traces, layout=layout)
    # Show the figure
    fig.show()

# clustering plot ____________________________________________________________________________________________________________________________
def score_plot(methods, optimal_ks):
    words = [s.name for s in methods]
    subplots_data = [] 
    for i, data in enumerate(methods):
        line_trace = go.Scatter(x=data.index, y=data, mode='lines',
                                line=dict(color=SEVENSET[i%len(SEVENSET)]))
        marker_trace = go.Scatter(x=data.index, y=data, mode='markers', marker=dict(size=10,color=SEVENSET[i%len(SEVENSET)]),
                                  hovertemplate=" number of clusters (k): <b>%{x}</b><br>"+"score: <b>%{y}</b><br>"+"<extra></extra>")
        plot_trace = [line_trace, marker_trace]
        subplots_data.append(plot_trace)


    layout = go.Layout(width=1300,
        height=500,
        plot_bgcolor='#f7f7f7',
        paper_bgcolor="#f7f7f7")

    fig = make_subplots(rows=1, cols=len(methods), horizontal_spacing=0.1, shared_xaxes=True)

    for i, data in enumerate(subplots_data):
        for trace in data:
            fig.add_trace(trace, row=1, col=i+1)
            fig.update_xaxes(tickfont=dict(size= 14, family='calibri', color='black' ),
                             showline=True, linewidth=0.5, linecolor='black', mirror=True, dtick=1, row=1, col=i+1)
            fig.update_yaxes(title=dict(text=words[i]+" Score",
                                        font=dict(size= 18, color= 'black', family= "Calibri")),
                             tickfont=dict(size= 14, family='calibri', color='black' ),
                             showline=True, linewidth=0.5, linecolor='black', mirror=True, row=1, col=i+1)
        fig.add_vline(x=optimal_ks[i], line_width=1, line_dash="dash", line_color="black", row=1, col=i+1)
        
    fig.update_xaxes(title=dict(text="Number of Clusters (k)",
                                font=dict(size= 18, color= 'black', family= "Calibri")), row=1, col=2)

    # update the layout of the figure
    fig.update_layout(layout, showlegend=False) 

    fig.show()
    
# km function and scatter plot _______________________________________________________________________________________________________________
def scatter_cluster(n, df, columns):
    km = KMeans(n_clusters = n, n_init = 25, random_state = 1)
    km.fit(df)
    cluster_centers = pd.DataFrame(km.cluster_centers_, columns=df.columns)
    
    # predict the clusters to group the cryptocurrencies using the scaled data
    prediction = km.predict(df)
    
    # create the trace for the data points
    trace_points = go.Scatter(
        x=df[columns[0]],
        y=df[columns[1]],
        mode='markers',
        name='Coins',
        marker=dict(
            size=7.5,
            color=km.labels_,
            colorscale=SEVENSET,
            opacity=0.9,
            line=dict(
                width=1,
                color='black'
            )
        ),
        text=df.index,  # Set the hover text to the index value
        showlegend=False
    )

    # create the trace for the centroid points
    trace_centroids = go.Scatter(
        x=cluster_centers[columns[0]],
        y=cluster_centers[columns[1]],
        mode='markers',
        name='Cluster Centers',
        marker=dict(
            size=30,
            color=cluster_centers.index,
            colorscale=SEVENSET,
            symbol='circle',
            opacity=0.3,
            line=dict(
                width=1,
                color='black'
            )
        ),
        text=[f"Centroid {i}" for i in range(len(cluster_centers))],  # Set the hover text to "Centroid {i}"
        showlegend=False
    )

    # create dummy trace for legend
    dummy_point = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            size=7.5,
            color="lightgray",
            colorscale=SEVENSET,
            opacity=1,
            line=dict(
                width=1,
                color='black'
            )
            ),
        name="Coins"  # set the name to an empty string so it is not visible in the legend
    )
    
    dummy_centroids = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                size=30,
                color="lightgray",
                colorscale=SEVENSET,
                symbol='circle',
                opacity=1,
                line=dict(
                    width=1,
                    color='lightgray'
                )),
            name="Cluster Centers"  # set the name to a visible string so it appears in the legend
        )
    
    # define the layout of the plot
    layout = go.Layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor= '#ffffff',
            font=dict(color='black', size=14)

    ),
        width=700,
        height=700,
        title=dict(text="Clustering with k= "+str(n),
                  font=dict(size= 20, color= 'black', family= "Times New Roman"),
                  x=0.5,
                  y=0.9),
        xaxis=dict(title='Price Change Percentage 24h',
                  showline=True,
            linewidth=0.5,
            linecolor='black',
            mirror=True,
                  color= 'black',
                   gridcolor='white'),
        yaxis=dict(title='Price Change Percentage 7d',
                   showline=True,
                   linewidth=0.5,
                   linecolor='black',
                   mirror=True,
                   color= 'black',
                   gridcolor='white'),
        hovermode='closest',
        plot_bgcolor='#ffffff',
        paper_bgcolor="#f7f7f7"
    )

    # create the figure object and add the traces to it
    fig = go.Figure(data=[trace_points, trace_centroids, dummy_point, dummy_centroids], layout=layout)
    
    # Show the figure
    return fig.show(), prediction

# km function and 3dscatter plot _____________________________________________________________________________________________________________
def scatter_3d_cluster(n, df, columns):
    km = KMeans(n_clusters = n, n_init = 25, random_state = 1)
    km.fit(df)
    cluster_centers = pd.DataFrame(km.cluster_centers_, columns=df.columns)
    
    # predict the clusters to group the cryptocurrencies using the scaled data
    prediction = km.predict(df)
    
    # create the trace for the data points
    trace_points = go.Scatter3d(
        x=df[columns[0]],
        y=df[columns[1]],
        z=df[columns[2]],
        mode='markers',
        name='Coins',
        marker=dict(
            size=4,
            color=km.labels_,
            colorscale=SEVENSET,
            opacity=0.9,
            line=dict(
                width=1,
                color='black'
            )
        ),
        text=df.index,  # Set the hover text to the index value
        showlegend=False
    )
    
    # create the trace for the centroid points
    trace_centroids = go.Scatter3d(
        x=cluster_centers[columns[0]],
        y=cluster_centers[columns[1]],
        z=cluster_centers[columns[2]],
        mode='markers',
        name='Cluster Centers',
        marker=dict(
            size=15,
            color=cluster_centers.index,
            colorscale=SEVENSET,
            symbol='circle',
            opacity=0.3,
            line=dict(
                width=1,
                color='black'
            )
        ),
        text=[f"Centroid {i}" for i in range(len(cluster_centers))],  # Set the hover text to "Centroid {i}"
        showlegend=False
    )
    
    # define the layout of the plot
    layout = go.Layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor= '#ffffff',
            font=dict(color='black', size=14)

    ),
        width=750,
        height=750,
        title=dict(text="Clustering with k= "+str(n),
                  font=dict(size= 20, color= 'black', family= "Times New Roman"),
                  x=0.5,
                  y=0.9),
       scene=dict(
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1),
        xaxis=dict(title=columns[0],
                  showline=True,
                  linewidth=0.5,
                  linecolor='black',
                  mirror=True,
                  color= 'black',
                  gridcolor='darkgray',
                  showbackground=True, # set the x-axis background to be visible
                  backgroundcolor="#ffffff"), # set the x-axis background color to white,
        yaxis=dict(title=columns[1],
                   showline=True,
                   linewidth=0.5,
                   linecolor='black',
                   mirror=True,
                   color= 'black',
                   gridcolor='darkgray',
                   showbackground=True, # set the x-axis background to be visible
                  backgroundcolor="#ffffff"), # set the x-axis background color to white,
        zaxis=dict(title=columns[2],
                   showline=True,
                   linewidth=0.5,
                   linecolor='black',
                   mirror=True,
                   color= 'black',
                   gridcolor='darkgray',
                   showbackground=True, # set the x-axis background to be visible
                  backgroundcolor="#ffffff"), # set the x-axis background color to white,      
           bgcolor="#f7f7f7"
    ),
        hovermode='closest',
        paper_bgcolor="#f7f7f7"
    )

    # create the figure object and add the traces to it
    fig = go.Figure(data=[trace_points, trace_centroids], layout=layout)
    
    # Show the figure
    return fig.show(), prediction

print(f"☑ helpers is imporetd")    
# --------------------------------------------------------------------------------------------------------------------------------------------