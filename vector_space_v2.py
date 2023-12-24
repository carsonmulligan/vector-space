import json
import random
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Pre-trained model initialization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to generate synthetic company details
def generate_company_details():
    company = f"Company_{random.randint(100, 999)}"
    address = f"{random.randint(100, 999)} Fake St, City{random.randint(100, 999)}, Country"
    facility_canon_id = random.randint(1000, 9999)
    company_canon_id = random.randint(10000, 99999)
    return company, address, facility_canon_id, company_canon_id

# Synthetic goods descriptions with enriched details
descriptions = []
for i in range(1001, 1011):
    sender, sender_address, sender_facility_id, sender_company_id = generate_company_details()
    receiver, receiver_address, receiver_facility_id, receiver_company_id = generate_company_details()
    transaction_date = f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/2023"
    
    descriptions.append({
        "transaction_id": i,
        "goods_description": f"Goods description {i}",
        "sender_company": sender,
        "sender_address": sender_address,
        "sender_facility_canon_id": sender_facility_id,
        "sender_company_canon_id": sender_company_id,
        "receiver_company": receiver,
        "receiver_address": receiver_address,
        "receiver_facility_canon_id": receiver_facility_id,
        "receiver_company_canon_id": receiver_company_id,
        "transaction_date": transaction_date
    })

# Vectorizing descriptions
vectors = []
for item in descriptions:
    vector = model.encode(item["goods_description"])
    vectors.append({key: item[key] for key in item if key != "goods_description"})
    vectors[-1]["vector"] = vector.tolist()

# Save vectors to a file
with open('enriched_vectors.json', 'w') as outfile:
    json.dump(vectors, outfile)

print("Vectorization and enrichment complete. Data saved to 'enriched_vectors.json'")
