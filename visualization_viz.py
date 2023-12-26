import json
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go

# Load vectors from the file
with open('enriched_vectors.json', 'r') as file:
    data = json.load(file)
vectors = [item["vector"] for item in data]

# Convert list of vectors into a 2D array
X = np.array(vectors)

# Check the number of samples and adjust perplexity if needed
n_samples = len(X)
perplexity = min(30, n_samples - 1)  # Default perplexity is 30, adjust if you have fewer samples

# Perform t-SNE with adjusted perplexity
tsne = TSNE(n_components=3, perplexity=perplexity, random_state=0)
X_reduced = tsne.fit_transform(X)

# Extracting individual coordinates for plotting
x, y, z = X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2]

# Create a 3D scatter plot
trace = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=5,
        opacity=0.8,
    )
)

# Define the layout of the plot
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis=dict(title='Component 1'),
        yaxis=dict(title='Component 2'),
        zaxis=dict(title='Component 3'),
    )
)

fig = go.Figure(data=[trace], layout=layout)
fig.show()
