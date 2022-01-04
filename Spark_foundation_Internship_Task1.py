#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")


# In[3]:


s_data.head(10)


# In[4]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[5]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# In[6]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# In[11]:


### **Training the Algorithm**
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print("Training complete.")


# In[12]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_


# In[13]:


# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[15]:


### **Making Predictions**
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test)


# In[16]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[21]:


# You can also test with your own data
own_pred = regressor.predict([[9.25]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[23]:


### **Evaluating the model**
from sklearn import metrics 
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




