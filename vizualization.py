import json
import numpy as np
import plotly.graph_objs as go

# Load vectors from the file
with open('vectors.json', 'r') as infile:
    vectors = json.load(infile)

# Extracting the vectors and transaction IDs
vec_array = np.array([item['Vector'] for item in vectors])
transaction_ids = [item['TransactionID'] for item in vectors]

# Using PCA for dimensionality reduction to 3D
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
vecs_reduced = pca.fit_transform(vec_array)

# Create a scatter plot
trace = go.Scatter3d(
    x=vecs_reduced[:, 0],
    y=vecs_reduced[:, 1],
    z=vecs_reduced[:, 2],
    mode='markers+text',
    marker=dict(
        size=5,
        color=transaction_ids,  # color points by TransactionID
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    ),
    text=transaction_ids,
    textposition="top center"
)

data = [trace]
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
fig = go.Figure(data=data, layout=layout)

# Render the plot
fig.show()
