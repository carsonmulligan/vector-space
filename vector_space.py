import json
from sentence_transformers import SentenceTransformer

# Pre-trained model initialization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Synthetic goods descriptions
descriptions = [
    {"TransactionID": 1001, "Description": "High-Precision Silicon Wafers for Microprocessor Fabrication"},
    {"TransactionID": 1002, "Description": "Integrated Circuit Grade Gallium Arsenide Substrates"},
    {"TransactionID": 1003, "Description": "Low-Noise Amplifier Chips for RF Applications"},
    {"TransactionID": 1004, "Description": "High-Efficiency Photovoltaic Cells for Energy Harvesting"},
    {"TransactionID": 1005, "Description": "Multi-Core Processing Units for High-Throughput Computing"},
    {"TransactionID": 1006, "Description": "Advanced Microcontroller Units for Embedded Systems"},
    {"TransactionID": 1007, "Description": "Quantum Dot LEDs for High-Resolution Displays"},
    {"TransactionID": 1008, "Description": "Nanoimprint Lithography Templates for Circuit Design"},
    {"TransactionID": 1009, "Description": "Flexible Printed Circuit Boards for Wearable Technology"},
    {"TransactionID": 1010, "Description": "Thermal Interface Materials for Heat Dissipation"}
]

# Vectorizing descriptions
vectors = []
for item in descriptions:
    vector = model.encode(item["Description"])
    vectors.append({"TransactionID": item["TransactionID"], "Vector": vector.tolist()})

# Save vectors to a file
with open('vectors.json', 'w') as outfile:
    json.dump(vectors, outfile)

print("Vectorization complete. Vectors saved to 'vectors.json'")
