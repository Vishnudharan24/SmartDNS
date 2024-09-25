import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from collections import Counter
import tldextract
import math
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to calculate the entropy of a domain
def calculate_entropy(domain):
    p, lns = Counter(domain), float(len(domain))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())

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

# Load the dataset from a CSV file (assuming it's called 'balanced_urls.csv')
df = pd.read_csv('balanced_urls.csv')

# Extract features from the 'url' column (instead of 'Domain')
features_list = []
for domain in df['url']:  # Changed 'Domain' to 'url'
    features_list.append(extract_features(domain))

# Convert features to a DataFrame
df_features = pd.DataFrame(features_list)

# Label encoding the 'label' column ('benign' and 'malicious')
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])  # Encode benign: 0, malicious: 1

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Split dataset into training and testing sets
X = df_features  # Features extracted from domains
y = df['label_encoded']  # Encoded labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple feedforward neural network
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (0 for benign, 1 for malicious)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to an .h5 file
model.save('dns_filter_model.h5')
print("Model saved as 'dns_filter_model.h5'.")

# Optional: Feature Importance (for Random Forest)
# Uncomment if you want to use Random Forest instead of NN
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
# importances = rf_model.feature_importances_
# feature_names = X.columns
# important_features = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
# print("\nFeature Importances:")
# print(important_features)