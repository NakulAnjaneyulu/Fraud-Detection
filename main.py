# main.py

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the data
data = pd.read_csv("C:/Project/Fraud detection/details.csv")

# Display first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for null values
print("\nChecking for null values:")
print(data.isnull().sum())

# Exploring transaction type
type_counts = data["type"].value_counts()
transactions = type_counts.index
quantity = type_counts.values

# Plotting the distribution of transaction types
figure = px.pie(data, values=quantity, names=transactions, hole=0.5, title="Distribution of Transaction Type")
figure.show()

# Checking correlation
correlation = data.corr()
print("\nCorrelation with 'isFraud':")
print(correlation["isFraud"].sort_values(ascending=False))

# Mapping transaction types and fraud labels
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

# Display updated dataset
print("\nUpdated dataset:")
print(data.head())

# Splitting the data into features and labels
X = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

# Splitting the dataset into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=42)

# Training a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
score = model.score(xtest, ytest)

# Display the accuracy of the model
print("\nModel accuracy:", score)

# Predicting a single instance
features = np.array([[4, 9000.60, 9000.60, 0.0]])
prediction = model.predict(features)
print("\nPrediction for features", features, ":", prediction)
