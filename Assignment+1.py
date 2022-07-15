
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[62]:


import pandas as pd

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)
df.head(10)


# In[83]:


def date_sorter():
    import re
    import datetime as dt
    import numpy as np
    import pandas as pd

    df = pd.Series(doc)
    
    # Your code here
    dateStr = []

    for i in range(len(df)):
        text = df.loc[i].replace(', ',' ').replace('. ',' ').replace('-','/')

        # 04/20/2009 4/20/2009 04/20/09  4/20/09 04/20/09  4/20/09
        re1 = r'\d{1,2}[/]\d{1,2}[/]\d{2,4}'

        # 'Mar/20/2009; Mar 20, 2009; March 20 2009; Mar 20 2009;Mar 25th, 2009; Mar 21st, 2009; Mar 22nd, 2009'
        re2 = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[/ ]\d{1,2}[/ ]\d{2,4}'

        # 20 Mar 2009; 20 March 2009; 20 Mar 2009; 20 March 2009; 28/June/2002, 
        re3 = r'\d{1,2}[/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[/ ]\d{2,4}'

        # Feb 2009; Sep 2009; Oct 2010; Jan/2021 July 1990
        re4 = r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[/ ]\d{4}'

        # 6/2008; 12/2009; 2009; 2010
        re5 = r'(?:\d{0,2})(?:/|)\d{4}'

        if re.search(re1, text):
            t = re.findall(re1, text)[0]
            yr = t.split('/')[2]                       
            daymonth = t.split('/')[0:2]
            daymonth.append(yr if len(yr)==2 else yr[-2:])
            t = '/'.join(daymonth)                      
            dateStr.append(dt.datetime.strptime(t,"%m/%d/%y").timestamp())

        elif re.search(re2, text):
            t = re.findall(re2, text)[0].replace('st','').replace('nd','').replace('rd','').replace('th','')
            t = t.replace(' ','/')
            dateStr.append(dt.datetime.strptime(t.replace(t.split('/')[0], t.split('/')[0][0:3]),"%b/%d/%Y").timestamp())

        elif re.search(re3, text):
            t = re.findall(re3, text)[0].replace('st','').replace('nd','').replace('rd','').replace('th','')
            t = t.replace(' ','/')
            dateStr.append(dt.datetime.strptime(t.replace(t.split('/')[1], t.split('/')[1][0:3]),"%d/%b/%Y").timestamp())

        elif re.search(re4,text):

            t = re.findall(re4, text)[0].replace(' ','/')
            yr = t.split('/')[1]
            month = t.split('/')[0]
            yr = yr if len(yr)==2 else yr[-2:]
            t = '/'.join([t.split('/')[0][0:3], yr])
            dateStr.append(dt.datetime.strptime(t,"%b/%y").timestamp())

        elif re.search(re5, text):

            t = re.findall(re5, text)[0].replace('-','/')
            t = t if '/' in t else ''.join(['01/',t])
            dateStr.append(dt.datetime.strptime(t,"%m/%Y").timestamp())
    y = pd.Series(np.argsort(dateStr).tolist())
    return y  # Your answer here


# In[ ]:




