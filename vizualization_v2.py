import json
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

# Load vectors from the file
with open('vectors.json', 'r') as infile:
    vectors = json.load(infile)

# Extracting the vectors and descriptions
vec_array = np.array([item['Vector'] for item in vectors])
descriptions = [item['Description'] for item in vectors]  # Updated to descriptions

# Using PCA for dimensionality reduction to 3D
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
        color=np.linspace(0, 1, len(descriptions)),  # Continuous color scale
        colorscale='Viridis',
        opacity=0.8
    ),
    text=descriptions,  # Use descriptions as text labels
    textposition="top center"
)

data = [trace]
layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
fig = go.Figure(data=data, layout=layout)

# Render the plot
fig.show()
