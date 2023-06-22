#!/usr/bin/env python
# coding: utf-8

# ##### Importing Required Libraries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data_set = pd.read_csv("C:\\Users\\rutuj\\OneDrive\\Documents\\Unemployment_Rate_upto_11_2020.csv")
data_set.head(5)


# In[5]:


data_set=data_set.rename(columns={data_set.columns[0]:'State',data_set.columns[3]:'EUR(%)',data_set.columns[4]:'EE',data_set.columns[5]:'ELPR(%)',data_set.columns[6]:'Region'})
data_set


# In[6]:


# Checking column names
data_set.columns


# In[7]:


data_set.describe()


# ##### Checking for Null Values:

# In[8]:


data_set.isnull().sum()


# In[9]:


data_set['State'].unique()


# In[10]:


data_set['Region'].unique()


# In[11]:


data_set.groupby("Region").size()


# In[12]:


reg_stat=data_set.groupby(['Region'])[['EUR(%)','EE', 'ELPR(%)']].mean().reset_index()
region_stat=round(reg_stat,2)
region_stat


# ##### Visualization Using Heat Map:

# In[13]:


heat_map=data_set[['EUR(%)','EE', 'ELPR(%)', 'longitude', 'latitude']]
heat_map=heat_map.corr()
plt.figure(figsize=(9,5))
sns.set_context('notebook',font_scale=0.75)
sns.heatmap(heat_map,annot=True,cmap='crest');


# ##### Unemployment Rate for different regions in India:

# In[14]:


data_set.columns=['State', 'Date', 'Frequency', 'EUR(%)','EE', 'ELPR(%)','Region', 'longitude', 'latitude']
plt.figure(figsize=(9,5))
plt.title("Unemployment Rate according to different regions")
sns.histplot(data=data_set,x="EUR(%)",hue="Region",stat='count',multiple="stack")
plt.show()


# ##### Unemployment Rate in every State and Region:

# In[16]:


import plotly.express as px
data=data_set[["State","Region","EUR(%)"]]
fig=px.sunburst(data,path=["Region","State"],values="EUR(%)", title="Unemployment Rate in every State and Region",color='EUR(%)',height=600)
fig.show()


# ##### Average Unemployment Rate for Region:

# In[17]:


fig=px.bar(reg_stat,x='Region',y='EUR(%)',title='Average Unemployment Rate for Region',color='Region',text_auto=True)
fig.update_layout(xaxis={'categoryorder':'total ascending'})
fig.show()


# ##### Thank You

# In[ ]:




