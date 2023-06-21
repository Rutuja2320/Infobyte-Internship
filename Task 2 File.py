#!/usr/bin/env python
# coding: utf-8

# ##### Importing required libraries:

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[2]:


data= pd.read_csv("C:\\Users\\rutuj\\OneDrive\\Documents\\Advertising.csv")
data.head(5)


# In[3]:


data.columns


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull()


# ##### Visualizing Correlation by using Heatmap:

# In[7]:


plt.figure(figsize=(9,5))
sns.heatmap(data.corr(),annot=True)


# ##### Seperate features and target 

# In[8]:


data1 = data.values
x=data1[:,0:3]
y=data1[:,3:]


# ##### Split Data Into Train and Test datasets

# In[9]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=0,test_size=0.3)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# ##### Linear Regression Model:

# In[10]:


lin_reg=LinearRegression()
lin_reg.fit(X_train,Y_train)


# In[11]:


pred1=lin_reg.predict(X_test)
pred1


# In[12]:


# Regression coefficients and intercept of the model
print(lin_reg.coef_, lin_reg.intercept_)


# In[13]:


print("Mean Square Error of Linear Regression model=",mse(Y_test,pred1)**0.5)
print("Coefficient of determination of Linear Regression model=",r2_score(Y_test,pred1))


# ##### Random Forest Regression:

# In[14]:


ran_forest=RandomForestRegressor()
ran_forest.fit(X_train,Y_train)


# In[16]:


pred2=ran_forest.predict(X_test)
pred2


# In[17]:


print("Mean Square Error of Random Forest Regression model",mse(Y_test,pred2)**0.5)
print("Coefficient of determination of Linear Regression model=",r2_score(Y_test,pred2))


# ##### Conclusion:
# #####                     The model which has less RMSE and more r2 score is better model, So from the above the Random Forest Regression is better model than the Linear Regression model. 

# ##### Thank you

# In[ ]:




