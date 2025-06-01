#!/usr/bin/env python
# coding: utf-8

# In[1]:


# loading libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[2]:


# data wrangling and setting features and labels

df = pd.read_csv('data.csv')

x = np.array(df[["x (choice)"]])
y = np.array(df["y (pos)"])


# In[3]:


# Logistic Regression and Model Fit

model = LogisticRegression(random_state = 42) # The answwer to life, universe and everything

model.fit(x, y)


# In[4]:


# input

print("For the positions:\n\n1\t\t2\t\t3\n\n\t4\t\t5\n\n")


# In[5]:


# prediction

make_perc = lambda i: i * 100

for i in range (1, 6):
    print(f"For {i}:")
    pick = model.predict([[i]])
    prob = model.predict_proba([[i]])


    prob = np.round(make_perc(prob), 2)

    # print(f"{prob[0]}")
    print(f"\n\nProbabilities based off of position:\n\n{prob[0][0]}\t\t{prob[0][1]}\t\t{prob[0][2]}\n\n\t{prob[0][3]}\t\t{prob[0][4]}\n\n")

    print(f"You should pick {pick[0]}\n\n")


# In[ ]:


### for executable script (not important)

print("Press any key to close")
input()


# In[ ]:




