#!/usr/bin/env python
# coding: utf-8

# ##### Importing Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# In[2]:


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species'] 


# In[3]:


df=pd.read_csv(r'C:/Users/rutuj/Downloads/iris-flower-classification-project/iris.data',header=None,names=columns)
df.head(5)


# In[4]:


df.info()


# In[5]:


df['Species'].value_counts()


# In[6]:


df.describe()


# ##### Data Visualization

# In[7]:


#Pair Plot: (It shows distinctive relationships between attributes)
sns.pairplot(df,hue='Species',height=3,aspect=1)
plt.show()


# In[9]:


#Box Plot: (It used to understand various measures such as max, min, mean, median and deviation.)
df.boxplot(by='Species',figsize=(13,10))
plt.show()


# ##### Seperate features and target 

# In[8]:


data = df.values
X = data[:,0:4]
Y = data[:,4]


# ##### Split Data Into Train and Test datasets

# In[10]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.3)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# ##### Logistic Regression:

# In[11]:


logis_reg = LogisticRegression()


# In[12]:


logis_reg.fit(X_train,Y_train)


# In[30]:


pred1=logis_reg.predict(X_test)
pred1


# In[13]:


print('Accuracy Score of Logistic Regression: ',logis_reg.score(X_test,Y_test))


# ##### Random Forest:

# In[23]:


randomfor = RandomForestClassifier(n_estimators=4) 


# In[24]:


randomfor.fit(X_train,Y_train)


# In[31]:


pred2=randomfor.predict(X_test)
pred2


# In[25]:


print('Accuracy Score of Random Forest: ',randomfor.score(X_test,Y_test))


# ###### Classification Report For Logistic Regression model:

# In[34]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, pred1))


# ##### Classification Report For Random Forest Model:

# In[35]:


print(classification_report(Y_test, pred2))


# #### Thank You

# In[ ]:




