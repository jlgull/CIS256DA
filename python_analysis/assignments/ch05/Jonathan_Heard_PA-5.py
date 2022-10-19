#!/usr/bin/env python
# coding: utf-8

# # Project Assignment 5 - Get different types of data 
# ## Author: Jonathan Heard
# ## Class: CIS256DA

# Chapter 5 Asignment
# 
# Search KaggleLinks to an external site. for something that you are interested in to see if there is any datasets related to that interest. If there is, download that dataset.  If not, just download one of the trending datasets in the trending section as shown below (2 pts.):
# 
# Tasks
# 
# 1. Read the data from the CSV file into a DataFrame (2 pts.).
# 2. Display the first five rows of data (2 pts.).
# 3. Display information about each of the columns (2 pts.).
# 
# Question
# 
# Why did you choose the downloaded dataset (2 pts.)?
# 
# 
# After completing the assignment, please submit your .ipynb file and the dataset you downloaded from Kaggle.
# 
# 

# ### The url for the site the data came from is:
# 
# https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
# 
# The download was a 9 MB zip file, 
#     called "Malicious_and_Phishing_attacks_ulrs_archive.zip"

# ## Import required modules

# In[1]:


# Import the modules needed for this project.

# Import pandas and assign the alias pd
import pandas as pd

# Import ZipFile from the zipfile module
from zipfile import ZipFile


# ## Tasks list

# In[2]:


# Extract the files from the zip file and diplsy the file names

file_names = list()
with ZipFile('Malicious_and_Phishing_attacks_ulrs_archive.zip', mode = 'r') as zip:
    zip.extractall()
    for file in zip.infolist():
        file_names.append(file.filename)
        print(file.filename, file.compress_size, file.file_size)
        


# In[3]:


# Task 01 Read the data from the CSV file into a DataFrame (2 pts.).

phishing = pd.read_csv('phishing_site_urls.csv')



# In[4]:


# Task 02 Display the first five rows of data (2 pts.).

phishing.head()


# In[5]:


# Task 03 Display information about each of the columns (2 pts.).

phishing.info()


# ## Question

# ### Why did you choose the downloaded dataset (2 pts.)?
# 
# The reason for my line of inquiry, is that I am always looking into Cybersecurity topics. I was not very happy with
#     my first few attempts at this assignment.
# 
# I had to download 4 different zip files, before I found one that would let step 1 proceed.
#     "Read the data from the CSV file into a DataFrame."
# 
# I finally decided to quit looking once, I was able to satisfy the requirements for this assignment.
# 
# The "bad" csv files I tried, all gave the same error messages as in the Jonathan_Heard_PA-5_Bad_csv.ipynb. I am 
#     include
#     
# 
# 

# In[ ]:




