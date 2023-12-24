from node2vec import Node2Vec
import networkx as nx
import json

# Load the enriched data
with open('enriched_vectors.json', 'r') as infile:
    data = json.load(infile)

# Create a graph
G = nx.Graph()

# Add nodes and edges
for transaction in data:
    G.add_node(transaction["sender_company"], type='company')
    G.add_node(transaction["receiver_company"], type='company')
    G.add_edge(transaction["sender_company"], transaction["receiver_company"])

# Initialize Node2Vec model
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Train the model
model = node2vec.fit(window=10, min_count=1)

# Get the vector for a node (company)
vector_for_company1 = model.wv['Company_539']

# Use the embeddings as needed for analysis or machine learning
