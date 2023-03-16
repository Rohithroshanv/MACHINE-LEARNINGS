#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_breast_cancer


# In[2]:


cancer=load_breast_cancer()


# In[3]:


cancer.keys()


# In[4]:


print(cancer['DESCR'])


# In[5]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[6]:


cancer['target']
print(cancer['target_names'])


# In[7]:


x=df
y=cancer['target']


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_valid_test,y_train,y_valid_test=train_test_split(x,y,test_size=0.3)
x_valid,x_test,y_valid,y_test=train_test_split(x_valid_test,y_valid_test,test_size=0.5)
print(len(x_train),len(x_valid),len(x_test))


# In[9]:


from sklearn.svm import SVC
#SVC support vector classifier
model=SVC()
model.fit(x_train,y_train)


# In[10]:


prediction=model.predict(x_test)


# In[11]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))


# In[12]:


prediction1=model.predict(x_valid_test)


# In[13]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_valid_test,prediction1))
print(confusion_matrix(y_valid_test,prediction1))


# In[14]:


# Grid search
from sklearn.model_selection import GridSearchCV
#C control the misclassification and gamma < large the varaiance
param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.1,0.001,0.0001]}
#grid=GridSearchCV(SVC(kernal='Linear'),param_grid,verbose=3)
grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid


# In[15]:


grid.fit(x_train,y_train)


# In[16]:


grid.best_params_


# In[17]:


pred=grid.predict(x_test)


# In[18]:


pred1=grid.predict(x_valid_test)


# In[19]:


print(classification_report(y_test,pred))


# In[20]:


print(classification_report(y_valid_test,pred1))


# In[21]:


print(confusion_matrix(y_test,pred))


# In[22]:


print(confusion_matrix(y_valid_test,pred1))


# In[23]:


grid.score(x_test,y_test)


# In[24]:


grid.score(x_valid_test,y_valid_test)


# In[25]:


grid.score(x_train,y_train)


# In[26]:


#normalise the dataset standardscaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
sc.fit(x_test)
x_test_std=sc.transform(x_test)


# In[27]:


#normalise the dataset standardscaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
sc.fit(x_test)
x_test_std=sc.transform(x_valid_test)


# In[28]:


grid.fit(x_train_std,y_train)


# In[29]:


pred=grid.predict(x_test_std)


# In[31]:


classification_report(y_valid_test,pred1)


# In[32]:


classification_report(y_valid_test,pred1)


# In[35]:


confusion_matrix(y_valid_test,pred)


# In[36]:


confusion_matrix(y_valid_test,pred1)


# In[38]:


from sklearn import metrics
metrics.accuracy_score(y_valid_test,pred1)


# In[39]:


metrics.precision_score(y_valid_test,pred1)


# In[40]:


metrics.f1_score(y_valid_test,pred1)


# In[41]:


df.shape


# In[ ]:




