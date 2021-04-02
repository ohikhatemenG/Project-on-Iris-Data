#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv(url, names=names)


# In[11]:


data


# In[13]:


data.shape


# In[14]:


data.head(10)


# In[15]:


data.info()


# In[16]:


data.isna().sum()


# In[17]:


data.describe()


# In[18]:


data.groupby('class').size()


# In[23]:


data.plot(kind='box', subplots=True, layout=(2,2), sharey=False, sharex=False,)
plt.show()


# In[24]:


data.hist()
plt.show()


# In[26]:


from pandas.plotting import scatter_matrix
scatter_matrix(data)


# In[30]:


array = data.values
X = array[:,0:4]
y = array[:, 4]
x_train, x_validation,y_train,y_validation = train_test_split(X,y, test_size=25, random_state=1)
# Build model
models = []
models.append(('LG', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GB', GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
# evaluate each model
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state = 1, shuffle=True)
    cv_results = cross_val_score(model,x_train,y_train, cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' %(name, cv_results.mean(), cv_results.std()))


# In[32]:


# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('COMPARE ALGORITHMS')
plt.show()


# In[34]:


# Make prediction, on validation datasets
model = SVC(gamma='auto')
model.fit(x_train, y_train)
yhat = model.predict(x_validation)
print(yhat)
# evaluate the prediction
print(accuracy_score(y_validation, yhat))
print(confusion_matrix(y_validation, yhat))
print(classification_report(y_validation, yhat))


# In[ ]:




