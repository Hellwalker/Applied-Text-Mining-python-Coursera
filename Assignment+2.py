
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 2 - Introduction to NLTK
# 
# In part 1 of this assignment you will use nltk to explore the Herman Melville novel Moby Dick. Then in part 2 you will create a spelling recommender function that uses nltk to find words similar to the misspelling. 

# ## Part 1 - Analyzing Moby Dick

# In[3]:


import nltk
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('inaugural')
nltk.download('nps_chat')
nltk.download('webtext')
nltk.download('treebank')
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


# ### Example 1
# 
# How many tokens (words and punctuation symbols) are in text1?
# 
# *This function should return an integer.*

# In[3]:


def example_one():
    
    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()


# ### Example 2
# 
# How many unique tokens (unique words and punctuation) does text1 have?
# 
# *This function should return an integer.*

# In[4]:


def example_two():
    
    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()


# ### Example 3
# 
# After lemmatizing the verbs, how many unique tokens does text1 have?
# 
# *This function should return an integer.*

# In[14]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()


# ### Question 1
# 
# What is the lexical diversity of the given text input? (i.e. ratio of unique tokens to the total number of tokens)
# 
# *This function should return a float.*

# In[15]:


def answer_one():
    
    return len(set(nltk.word_tokenize(moby_raw)))/len(nltk.word_tokenize(moby_raw))# Your answer here

answer_one()


# ### Question 2
# 
# What percentage of tokens is 'whale'or 'Whale'?
# 
# *This function should return a float.*

# In[20]:


def answer_two():
    
    return 100*(text1.count('whale')+text1.count('Whale'))/float(len(nltk.word_tokenize(moby_raw)))# Your answer here

answer_two()


# ### Question 3
# 
# What are the 20 most frequently occurring (unique) tokens in the text? What is their frequency?
# 
# *This function should return a list of 20 tuples where each tuple is of the form `(token, frequency)`. The list should be sorted in descending order of frequency.*

# In[48]:


def answer_three():
    from nltk.probability import FreqDist
    dist = FreqDist(moby_tokens)
    
    return dist.most_common(20)# Your answer here

answer_three()


# ### Question 4
# 
# What tokens have a length of greater than 5 and frequency of more than 150?
# 
# *This function should return an alphabetically sorted list of the tokens that match the above constraints. To sort your list, use `sorted()`*

# In[50]:


def answer_four():
    from nltk.probability import FreqDist
    dist = FreqDist(moby_tokens)
    vocab = dist.keys()
    freqwords = sorted([w for w in vocab if len(w) > 5 and dist[w]>150])
    return freqwords # Your answer here

answer_four()


# ### Question 5
# 
# Find the longest word in text1 and that word's length.
# 
# *This function should return a tuple `(longest_word, length)`.*

# In[57]:


def answer_five():
    from nltk.probability import FreqDist
    dist = FreqDist(text1)
    vocab = dist.keys()
    word = max(vocab, key= len)
    
    return (word, len(word))# Your answer here

answer_five()


# ### Question 6
# 
# What unique words have a frequency of more than 2000? What is their frequency?
# 
# "Hint:  you may want to use `isalpha()` to check if the token is a word and not punctuation."
# 
# *This function should return a list of tuples of the form `(frequency, word)` sorted in descending order of frequency.*

# In[66]:


def answer_six():
    from nltk.probability import FreqDist
    dist = FreqDist(moby_tokens)
    vocab = dist.keys()
    freqwords = sorted([(dist[w], w) for w in vocab if w.isalpha() and dist[w] > 2000], reverse = True)
    return freqwords# Your answer here

answer_six()


# ### Question 7
# 
# What is the average number of tokens per sentence?
# 
# *This function should return a float.*

# In[73]:


def answer_seven():
    
    sents = nltk.sent_tokenize(moby_raw)
    sum_sents  = 0
    for sent in sents:
        sum_sents+=len(nltk.word_tokenize(sent))
    
    return sum_sents/len(sents)# Your answer here

answer_seven()


# ### Question 8
# 
# What are the 5 most frequent parts of speech in this text? What is their frequency?
# 
# *This function should return a list of tuples of the form `(part_of_speech, frequency)` sorted in descending order of frequency.*

# In[97]:


def answer_eight():
    from collections import Counter
    nltk.download('averaged_perceptron_tagger')
    pos_token = nltk.pos_tag(moby_tokens)
    pos_freq = Counter([subl[1] for subl in pos_token]).most_common(5)
    
    return pos_freq# Your answer here

answer_eight()


# ## Part 2 - Spelling Recommender
# 
# For this part of the assignment you will create three different spelling recommenders, that each take a list of misspelled words and recommends a correctly spelled word for every word in the list.
# 
# For every misspelled word, the recommender should find find the word in `correct_spellings` that has the shortest distance*, and starts with the same letter as the misspelled word, and return that word as a recommendation.
# 
# *Each of the three different recommenders will use a different distance measure (outlined below).
# 
# Each of the recommenders should provide recommendations for the three default words provided: `['cormulent', 'incendenece', 'validrate']`.

# In[4]:


from nltk.corpus import words
nltk.download('words')

correct_spellings = words.words()


# ### Question 9
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the trigrams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[10]:


def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    from nltk.util import ngrams
    from nltk.metrics.distance import jaccard_distance as jd

    a = []
    b = []
    c = []
    for i in correct_spellings:
        # first capital should be the same
        a.append(jd(set(ngrams(list(i),3)),set(ngrams(list(entries[0]), 3))) if i[0] == entries[0][0] else 99999)
        b.append(jd(set(ngrams(list(i),3)),set(ngrams(list(entries[1]), 3))) if i[0] == entries[1][0] else 99999)
        c.append(jd(set(ngrams(list(i),3)),set(ngrams(list(entries[2]), 3))) if i[0] == entries[2][0] else 99999)
       
    # output jaccard distance minimum corresponding index and vocab word
    result = [correct_spellings[a.index(min(a))],
              correct_spellings[b.index(min(b))],
              correct_spellings[c.index(min(c))]]
    return result# Your answer here
    
answer_nine()


# ### Question 10
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) on the 4-grams of the two words.**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[11]:


def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    from nltk.util import ngrams
    from nltk.metrics.distance import jaccard_distance as jd
    
    n = 4

    a = []
    b = []
    c = []
    for i in correct_spellings:
        # first capital should be the same
        a.append(jd(set(ngrams(list(i),n)),set(ngrams(list(entries[0]), n))) if i[0] == entries[0][0] else 99999)
        b.append(jd(set(ngrams(list(i),n)),set(ngrams(list(entries[1]), n))) if i[0] == entries[1][0] else 99999)
        c.append(jd(set(ngrams(list(i),n)),set(ngrams(list(entries[2]), n))) if i[0] == entries[2][0] else 99999)
       
    # output jaccard distance minimum corresponding index and vocab word
    result = [correct_spellings[a.index(min(a))],
              correct_spellings[b.index(min(b))],
              correct_spellings[c.index(min(c))]]
    
    return result# Your answer here
    
answer_ten()


# ### Question 11
# 
# For this recommender, your function should provide recommendations for the three default words provided above using the following distance metric:
# 
# **[Edit distance on the two words with transpositions.](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)**
# 
# *This function should return a list of length three:
# `['cormulent_reccomendation', 'incendenece_reccomendation', 'validrate_reccomendation']`.*

# In[16]:


def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    from nltk.metrics.distance import edit_distance as ed

    a = []
    b = []
    c = []
    for i in correct_spellings:
        # first capital should be the same
        # True as transpositions = True
        a.append(ed(list(i),list(entries[0]), True) if i[0] == entries[0][0] else 99999)
        b.append(ed(list(i),list(entries[1]), True) if i[0] == entries[1][0] else 99999)
        c.append(ed(list(i),list(entries[2]), True) if i[0] == entries[2][0] else 99999)
       
    # output DL distance minimum corresponding index and vocab word
    result = [correct_spellings[a.index(min(a))],
              correct_spellings[b.index(min(b))],
              correct_spellings[c.index(min(c))]]
    return result # Your answer here 
    
answer_eleven()


# In[ ]:




