#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the dataset
file_path = 'C:\\Users\\Vamsi T\\Downloads\\USvideos.csv'
df = pd.read_csv(file_path)

# Check for missing values
df.isnull().sum()

# Summary statistics for numerical columns
df.describe()


# In[3]:


df.shape


# In[4]:


df = df.drop_duplicates()
df.shape


# In[5]:


df.info()


# In[6]:


columns_to_remove = ['thumbnail_link', 'description']
df = df.drop(columns= columns_to_remove)
df.info()


# In[7]:


from datetime import datetime
df['trending_date']= df['trending_date'].apply(lambda x: datetime.strptime(x, '%y.%d.%m'))
df['publish_time']= pd.to_datetime(df['publish_time'])
df.head()


# In[8]:


df['publish_month']= df['publish_time'].dt.month
df['publish_day']= df['publish_time'].dt.day
df['publish_hour']= df['publish_time'].dt.hour
df.head()


# In[9]:


import numpy as np
print(sorted(df['category_id'].unique()))
df['category_name']= np.nan
df.loc[(df['category_id'] == 1), 'category_name'] = 'Film and Animation'
df.loc[(df['category_id'] == 2), 'category_name'] = 'Autos and Vehicles'
df.loc[(df['category_id'] == 10), 'category_name'] = 'Music'
df.loc[(df['category_id'] == 15), 'category_name'] = 'Pets and Animals'
df.loc[(df['category_id'] == 17), 'category_name'] = 'Sports'
df.loc[(df['category_id'] == 19), 'category_name'] = 'Travel and Events'
df.loc[(df['category_id'] == 20), 'category_name'] = 'Gaming'
df.loc[(df['category_id'] == 22), 'category_name'] = 'People and Blogs'
df.loc[(df['category_id'] == 23), 'category_name'] = 'Comedy'
df.loc[(df['category_id'] == 24), 'category_name'] = 'Entertainment'
df.loc[(df['category_id'] == 25), 'category_name'] = 'News and Politics'
df.loc[(df['category_id'] == 26), 'category_name'] = 'How to and Style'
df.loc[(df['category_id'] == 27), 'category_name'] = 'Education'
df.loc[(df['category_id'] == 28), 'category_name'] = 'Science and Technology'
df.loc[(df['category_id'] == 29), 'category_name'] = 'Non Profits and Activism'
df.loc[(df['category_id'] == 30), 'category_name'] = 'Movies'
df.loc[(df['category_id'] == 43), 'category_name'] = 'Shows'

df.head()


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Bar plot: Top 10 videos by views
top_videos = df.groupby('title')['views'].sum().nlargest(10)
plt.figure(figsize=(10, 6))
plt.barh(top_videos.index, top_videos.values, color= 'skyblue')
plt.gca().invert_yaxis()
plt.title('Top 10 Videos by Views')
plt.xlabel('Views')
plt.ylabel('Video Title')
plt.xticks(rotation= 90)
plt.show()


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the matplotlib figure
plt.figure(figsize=(15, 15))

# Bar plot for category_id
plt.subplot(3, 1, 1)
sns.countplot(y='category_id', data=df, order=df['category_id'].value_counts().index)
plt.title('Distribution of Category ID')
plt.xlabel('Count')
plt.ylabel('Category ID')

# Bar plot for channel_title (top 20 channels by video count)
top_channels = df['channel_title'].value_counts().nlargest(20).index
plt.subplot(3, 1, 2)
sns.countplot(y='channel_title', data=df[df['channel_title'].isin(top_channels)], order=top_channels)
plt.title('Top 20 Channels by Video Count')
plt.xlabel('Count')
plt.ylabel('Channel Title')

# Bar plot for tags (top 20 tags by frequency)
df['tags'] = df['tags'].apply(lambda x: x.replace('"', '').split('|'))
all_tags = df['tags'].explode()
top_tags = all_tags.value_counts().nlargest(20).index
plt.subplot(3, 1, 3)
sns.countplot(y=all_tags[all_tags.isin(top_tags)], order=top_tags)
plt.title('Top 20 Tags by Frequency')
plt.xlabel('Count')
plt.ylabel('Tags')

plt.tight_layout()
plt.show()


# In[74]:


# Scatter plot: Views vs Likes
plt.figure(figsize=(10, 6))
sns.scatterplot(x='views', y='likes', data=df, alpha=0.5)
plt.title('Views vs Likes')
plt.xlabel('Views')
plt.ylabel('Likes')
plt.show()


# In[78]:


# Pie chart: Distribution of video categories
category_counts = df['category_name'].value_counts()

plt.figure(figsize=(10, 8))
colors = plt.get_cmap('tab20')(np.linspace(0.0, 1.0, len(category_counts)))

plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Distribution of Video Categories')

# Add a legend
plt.legend(category_counts.index, title="Categories", bbox_to_anchor=(1, 1), loc="best")

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[76]:


df['publish_date'] = df['publish_time'].dt.date

# Group by publish_date and sum views
views_trend = df.groupby('publish_date')['views'].sum().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(x='publish_date', y='views', data=views_trend)
plt.title('Trend of Views Over Time')
plt.xlabel('Date')
plt.ylabel('Total Views')
plt.xticks(rotation=45)
plt.show()


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the matplotlib figure
plt.figure(figsize=(15, 10))

# Distribution plots for key numerical variables
key_vars = ['views', 'likes', 'dislikes', 'comment_count']

for i, var in enumerate(key_vars, start=1):
    plt.subplot(2, 2, i)
    sns.histplot(df[var], bins=50, kde=True)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[45]:


# Distribution plots for key numerical variables
key_vars = ['views', 'likes', 'dislikes', 'comment_count']

# Set up the matplotlib figure for box plots
plt.figure(figsize=(15, 10))

for i, var in enumerate(key_vars, start=1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[var])
    plt.title(f'Box Plot of {var}')
    plt.xlabel(var)

plt.tight_layout()
plt.show()


# In[46]:


# Calculate correlations between numerical variables
correlation_matrix = df[key_vars].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[ ]:




