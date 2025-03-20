import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load sample dataset
data = pd.read_csv('C:/Users/Maithili/Desktop/GITHUB/ATM fraud detection/bank.csv')  # Replace with actual dataset

# Selecting relevant features for clustering
features = ["TransactionAmount", "TransactionDuration", "AccountBalance", "CustomerAge", "LoginAttempts"]
categorical_features = ["TransactionType", "Location", "Channel", "CustomerOccupation"]

# Preprocessing: One-hot encoding for categorical data & scaling numerical features
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Transform the dataset
X = preprocessor.fit_transform(data)

# Implementing three models for anomaly detection
# 1. Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
data['Anomaly_ISO'] = iso_forest.fit_predict(X)
data['Anomaly_ISO'] = data['Anomaly_ISO'].apply(lambda x: 1 if x == -1 else 0)

# 2. One-Class SVM
oc_svm = OneClassSVM(nu=0.02, kernel='rbf', gamma='auto')
data['Anomaly_SVM'] = oc_svm.fit_predict(X)
data['Anomaly_SVM'] = data['Anomaly_SVM'].apply(lambda x: 1 if x == -1 else 0)

# 3. Random Forest Classifier (Supervised for Comparison)
X_train, X_test, y_train, y_test = train_test_split(X, data['Anomaly_ISO'], test_size=0.2, random_state=42)
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

data['Anomaly_RFC'] = rfc.predict(X)

# Performance Comparison
print("Random Forest Classifier Performance:")
print(classification_report(y_test, y_pred))

# Visualizations
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['TransactionAmount'], y=data['AccountBalance'], hue=data['Anomaly_ISO'], palette={0: 'pink', 1: 'red'})
plt.xlabel("Transaction Amount")
plt.ylabel("Account Balance")
plt.title("Isolation Forest Anomaly Detection in ATM Withdrawals")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['TransactionAmount'], y=data['AccountBalance'], hue=data['Anomaly_SVM'], palette={0: 'pink', 1: 'red'})
plt.xlabel("Transaction Amount")
plt.ylabel("Account Balance")
plt.title("One-Class SVM Anomaly Detection in ATM Withdrawals")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['TransactionAmount'], y=data['AccountBalance'], hue=data['Anomaly_RFC'], palette={0: 'pink', 1: 'red'})
plt.xlabel("Transaction Amount")
plt.ylabel("Account Balance")
plt.title("Random Forest Classification in ATM Withdrawals")
plt.show()

# Boxplot for transaction amounts
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Anomaly_ISO'], y=data['TransactionAmount'], palette=['pink', 'red'])
plt.xlabel("Anomaly (0=Normal, 1=Anomalous)")
plt.ylabel("Transaction Amount")
plt.title("Transaction Amount Distribution by Anomaly")
plt.show()

# Save processed dataset
data.to_csv("processed_atm_transactions.csv", index=False)
