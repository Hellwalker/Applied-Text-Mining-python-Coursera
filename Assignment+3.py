
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[2]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[5]:


def answer_one():
    
    s = len(spam_data[spam_data.target==1])/len(spam_data)*100
    return s#Your answer here


# In[6]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[9]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vect = CountVectorizer().fit(X_train)
    return max(vect.get_feature_names(), key=len)#Your answer here


# In[10]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[13]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    vect = CountVectorizer().fit(X_train)
    model = MultinomialNB(alpha=0.1)
    model.fit(vect.transform(X_train),y_train)
    
    predictions = model.predict(vect.transform(X_test))
    return float(roc_auc_score(y_test, predictions))#Your answer here


# In[14]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[48]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    feature_names = np.array(vect.get_feature_names())
    tfidf_value_array = X_train_vectorized.max(0).toarray()[0]
    sorted_tfidf_index = tfidf_value_array.argsort()
    
    smallest_tfidfs_series = pd.Series(data = tfidf_value_array[sorted_tfidf_index[:20]], index = feature_names[sorted_tfidf_index[:20]])
    smallest_tfidfs_series = smallest_tfidfs_series.sort_index()
    
    largest_tfidfs_series = pd.Series(data = tfidf_value_array[sorted_tfidf_index[:-21:-1]], index = feature_names[sorted_tfidf_index[:-21:-1]])
    largest_tfidfs_series = largest_tfidfs_series.sort_index()
    
    return (smallest_tfidfs_series, largest_tfidfs_series)#Your answer here


# In[49]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[50]:


def answer_five():
    
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    
    predictions = model.predict(vect.transform(X_test))
    
    return float(roc_auc_score(y_test, predictions))#Your answer here


# In[51]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[60]:


def answer_six():
    
    a = spam_data[spam_data.target==1].text.apply(lambda x:len(x))
    b = spam_data[spam_data.target==0].text.apply(lambda x:len(x))
    
    return (np.mean(b), np.mean(a))#Your answer here


# In[61]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[62]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    # hstack of scipy.sparse
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[71]:


from sklearn.svm import SVC

def answer_seven():
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    model = SVC(C=10000)
    model.fit(add_feature(X_train_vectorized, X_train.apply(lambda x:len(x))), y_train)
    
    predictions = model.predict(add_feature(X_test_vectorized, X_test.apply(lambda x:len(x))))
    
    return float(roc_auc_score(y_test, predictions))#Your answer here


# In[72]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[83]:


def answer_eight():
    a = [sum(char.isnumeric() for char in x) for x in spam_data[spam_data.target==1].text]
    b = [sum(char.isnumeric() for char in x) for x in spam_data[spam_data.target==0].text]
    return (np.mean(b), np.mean(a))#Your answer here


# In[84]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[87]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    digit_per_doc_train = [sum(char.isnumeric() for char in x) for x in X_train]
    digit_per_doc_test = [sum(char.isnumeric() for char in x) for x in X_test]
    
    X_train_new = add_feature(add_feature(X_train_vectorized, X_train.apply(lambda x:len(x))), digit_per_doc_train)
    X_test_new = add_feature(add_feature(X_test_vectorized, X_test.apply(lambda x:len(x))), digit_per_doc_test)
    
    model = LogisticRegression(C=100)
    model.fit(X_train_new, y_train)
    
    predictions = model.predict(X_test_new)
    
    return float(roc_auc_score(y_test, predictions))#Your answer here


# In[88]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[91]:


import re

def answer_ten():
    a = [len(re.findall('\W', item)) for item in spam_data[spam_data.target==1].text]
    b = [len(re.findall('\W', item)) for item in spam_data[spam_data.target==0].text]
    return (np.mean(b),np.mean(a))#Your answer here


# In[92]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[99]:


def answer_eleven():
    
    vect = CountVectorizer(min_df = 5, analyzer = 'char_wb', ngram_range=(2,5)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    X_train_new = add_feature(X_train_vectorized, [X_train.str.len(),
                                                   X_train.apply(lambda x: len(''.join([a for a in x if a.isdigit()]))),
                                                   X_train.str.findall(r'(\W)').str.len()])
    X_test_new = add_feature(X_test_vectorized, [X_test.str.len(),
                                                 X_test.apply(lambda x: len(''.join([a for a in x if a.isdigit()]))),
                                                 X_test.str.findall(r'(\W)').str.len()])

    model = LogisticRegression(C=100)
    model.fit(X_train_new,y_train)
    
    sorted_coef_index = model.coef_[0].argsort()
    
    feature_names = np.array(vect.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
    
    smallest_coef_list = feature_names[sorted_coef_index[:10]]
    largest_coefs_list = feature_names[sorted_coef_index[:-11:-1]]
    
    predictions = model.predict(X_test_new)
    auc_score = float(roc_auc_score(y_test, predictions))
    return (auc_score, smallest_coef_list, largest_coefs_list)#Your answer here


# In[100]:


answer_eleven()


# In[ ]:




