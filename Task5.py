#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
file_path = 'C:\\Users\\Vamsi T\\Downloads\\heart.csv'
data = pd.read_csv(file_path)

# Feature Engineering
# Create age groups
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 75, 90], labels=['0-30', '31-45', '46-60', '61-75', '76-90'])

# Categorize cholesterol levels
data['chol_level'] = pd.cut(data['chol'], bins=[0, 200, 239, 500], labels=['Normal', 'Borderline High', 'High'])

# Interaction features
data['age_trestbps'] = data['age'] * data['trestbps']
data['age_chol'] = data['age'] * data['chol']
data['thalach_trestbps'] = data['thalach'] * data['trestbps']

# Convert categorical features to dummy variables
data = pd.get_dummies(data, columns=['age_group', 'chol_level'], drop_first=True)

# Correlation Analysis
plt.figure(figsize=(12, 10))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Prepare features for PCA
numeric_features = data.select_dtypes(include=[float, int]).drop(columns=['target'])

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['target'] = data['target']

# Plotting the PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='target', data=pca_df, palette='coolwarm')
plt.title('PCA of Heart Disease Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Explained variance
explained_variance = pca.explained_variance_ratio_
print('Explained variance by each component:', explained_variance)

# Feature Importance using Random Forest
X = data.drop(columns=['target'])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy:', accuracy)

# Get feature importances
feature_importances = rf.feature_importances_
features = X.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances')
plt.show()

