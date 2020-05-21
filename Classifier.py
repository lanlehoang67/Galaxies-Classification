#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from joblib import load
from PIL import Image
import xgboost as xgb
from sklearn.metrics import accuracy_score


# In[2]:


# read the solution data
solutions = pd.read_csv("data/solutions.csv")
# take only 10001 records due to memory restrictions
solutions = solutions.truncate(before=20000, after=29999, axis = "rows")
solutions.shape


# In[3]:


# image opener and flattener
def getImage(imgID):
    return np.array(Image.open("data/images_modified/" + str(imgID) + ".jpg")).flatten()

# open all the images
images = np.array([np.array(getImage(imgID)) for imgID in solutions["GalaxyID"]])
# normalize the pixels
images = images/255
images.shape


# In[4]:


solutions = solutions.loc[:, ["Smooth"]]
solutions.head(3)


# In[5]:


# load the ready model
model = load('xgboost.80')


# In[6]:


predictions = model.predict(images)


# In[ ]:


# test accuracy on the data not used for training
predictions = np.round(predictions)
accuracy_score(solutions, predictions)


# In[ ]:




