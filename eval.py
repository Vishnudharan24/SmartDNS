import pandas as pd
import tldextract
import math
import pickle
import tensorflow as tf
from collections import Counter  # Add this import for Counter

# Function to calculate the entropy of a domain
def calculate_entropy(domain):
    p, lns = Counter(domain), float(len(domain))
    return -sum(count / lns * math.log2(count / lns) for count in p.values())

# Function to extract features from domain names
def extract_features(domain):
    features = {}
    
    # Extract TLD, domain, and subdomain information
    ext = tldextract.extract(domain)
    
    # Domain length
    features['domain_length'] = len(ext.domain)
    
    # Number of subdomains
    features['num_subdomains'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
    
    # Length of full domain (with subdomains)
    features['full_domain_length'] = len(domain)
    
    # Entropy of domain name
    features['entropy'] = calculate_entropy(domain)
    
    # Domain has numbers
    features['has_numbers'] = any(char.isdigit() for char in ext.domain)
    
    # Domain has special characters
    features['has_special_chars'] = any(not char.isalnum() for char in ext.domain)
    
    # Subdomain length
    features['subdomain_length'] = len(ext.subdomain)
    
    # Total unique characters in domain
    features['unique_characters'] = len(set(ext.domain))
    
    return features

# Load the trained model and label encoder
model = tf.keras.models.load_model('dns_filter_model.h5')
print("Model loaded successfully.")

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("Label encoder loaded successfully.")

# Input for evaluation
domain = input("Enter the domain to evaluate (e.g., example.com): ")

# Extract features for the input domain
features = extract_features(domain)
print("\n=== Extracted Features ===")
print(features)
print("===========================")

# Convert features to a DataFrame for prediction
features_df = pd.DataFrame([features])

# Make prediction
prediction_prob = model.predict(features_df)[0][0]  # Get the predicted probability
predicted_label = 'malicious' if prediction_prob >= 0.5 else 'benign'

# Output the prediction result
print("\n=== Prediction Result ===")
print(f"Domain: {domain}")
print(f"Prediction Probability (Malicious): {prediction_prob:.4f}")
print(f"Predicted Label: {predicted_label}")
print("===========================")
