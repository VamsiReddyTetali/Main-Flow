#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from textblob import TextBlob

# Load the dataset
file_path = 'C:\\Users\\Vamsi T\\Downloads\\disney_plus_titles.csv'
disney_df = pd.read_csv(file_path)

# Convert 'date_added' to datetime format
disney_df['date_added'] = pd.to_datetime(disney_df['date_added'], errors='coerce')

# Drop rows with NaT in 'date_added'
disney_df = disney_df.dropna(subset=['date_added'])

# Extract year and month from 'date_added'
disney_df['year_added'] = disney_df['date_added'].dt.year.astype(int)
disney_df['month_added'] = disney_df['date_added'].dt.month.astype(int)

# Group by year and month to count titles added
titles_by_month = disney_df.groupby(['year_added', 'month_added']).size().reset_index(name='count')

# Create a date column by combining year and month with a fixed day value
titles_by_month['date'] = pd.to_datetime(titles_by_month['year_added'].astype(str) + '-' +
                                         titles_by_month['month_added'].astype(str) + '-01')

# Create a time series plot with the new date column
plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='count', data=titles_by_month)
plt.title('Number of Titles Added Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Titles')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sentiment Analysis on descriptions
disney_df['description'] = disney_df['description'].astype(str)
disney_df['sentiment'] = disney_df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Plot sentiment distribution
plt.figure(figsize=(12, 6))
sns.histplot(disney_df['sentiment'], bins=30, kde=True)
plt.title('Sentiment Distribution of Descriptions')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Text Mining and Clustering
# Using TF-IDF to vectorize descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(disney_df['description'])

# KMeans clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(tfidf_matrix)

# Assign clusters to titles
disney_df['cluster'] = kmeans.labels_

# PCA for visualization of clusters
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(tfidf_matrix.toarray())
colors = ['r', 'b', 'c', 'y', 'm']

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]

plt.figure(figsize=(12, 6))
for i in range(num_clusters):
    points = [k for k in range(len(x_axis)) if kmeans.labels_[k] == i]
    plt.scatter([x_axis[j] for j in points], [y_axis[j] for j in points], c=colors[i], label=f'Cluster {i}')
plt.title('KMeans Clusters of Descriptions')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

