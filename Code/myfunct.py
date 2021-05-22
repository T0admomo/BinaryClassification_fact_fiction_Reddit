#!/usr/bin/env python
# coding: utf-8

# In[17]:


from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ## Api Request's

# In[2]:


def get_posts(strokes, subreddit, target_time):
    df = pd.DataFrame()
    master_df = pd.DataFrame()
    
   
    for i in range(strokes):
        try:
            params = {
                'subreddit': subreddit,
                'size': 50,
                'before': target_time}

            res = requests.get(url,params)
            data = res.json()
            posts = data['data']
            df = pd.DataFrame(posts)

            frames = [df,master_df]
            master_df = pd.concat(frames, axis= 0 , ignore_index = True)

            target_time = df['created_utc'].min()

            time.sleep(10)
        except KeyError:
            continue

    
# for when created_utc does not exist     

       
    return master_df


# In[3]:


# Creates dataframe from an HTML request response
def res_to_df(res):
    data = res.json()
    posts = data['data']
    return pd.DataFrame(posts)


# ## Visualizations

# In[4]:


def disp_hud(hud):
    base_group = corpus.groupby(['fact']).mean()
    head = corpus.head(2)
    
    hud = [base_group,head]
    disp = ['mean','preview']
    
    for i,li in enumerate(hud):
        print(disp[i])
        display(li)


# In[ ]:


def disp_cm(model,X,y):
    preds = model.predict(X)
   
    cm = confusion_matrix(y, preds)
    
    fig  = ConfusionMatrixDisplay(cm).plot();
    fig = plt.title(f'{model[1]} \n MSE: {mean_squared_error(y ,preds)}')
    fig = plt.xlabel((('Fiction'),('Fact')))
    
    return fig


# ## Cleaning & EDA

# ## Tokenizing 

# In[5]:


# tokenize the text field of each column and remove pnctuation.

def make_tokens(corpus,column):    
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    corpus[column] = [tokenizer.tokenize(row) for row in corpus[column]]
    return corpus


# In[6]:


# remove tokens from coprus with using text.ENGLISH_STOP_WORDS and custom stop word list 
def clean_tokens(corpus,column):    
    
    
    # custom stop word list
    my_stop_words = ['https','com','www','people','know','actually',
                     'world','time','years','fact','facts','fake','like',
                     'sk','10','en','day','water','did','just','the']
    
    # append custom stopwords to text.ENGLIS_STOP_WORDS
    stop_words = text.ENGLISH_STOP_WORDS.union(my_stop_words)
    
    # CLean rows of words in stop words list
    for row in corpus[column]:
        for word in row:
            if word in stop_words:
                row.remove(word)
    
    # returns clean datframe.
    return corpus


# In[7]:


def string_tokens(corpus,column):
    # returns tokinzed columns back to strings
    for i , row in enumerate(corpus[column]):
        corpus[column][i]=' '.join(corpus[column][i])
    # returns clean strings df
    return corpus


# In[8]:


def make_clean_string_corpus(corpus,column):
    # runs each of the three prior functions and returns a dataframe where the text field
    #   has had punctuation and stop words removed removed, and had been retunr to a string.
    
    corpus = make_tokens(corpus,column)
    corpus = clean_tokens(corpus, column)
    corpus = string_tokens(corpus,column)
            
    return corpus


# ## Model Fit and score 

# example list of pipelines requiered to run these functions
# 
# pipelines = [
# 
#     ("ModelName_1", pipeline_1),
#     
#     ("ModelName_2", pipeline_2),
# 
#     ]

# In[9]:


def fit_pipes(pipelines,X,y):
    """
     fits a model for each pipeline in our pipelines list
    """
    for name, model in pipelines:
        model.fit(X,y)


# In[15]:


def score_pipes(pipelines,X,y):
    """
    score all pipes in pipelines
    """
    scores = []
    df = pd.DataFrame()

    # for each name and model in pipelines
    for name, model in pipelines:
        # score the given population 
        scores.append(model.score(X,y))
        #print the madel name and score
        print( f'{name} Accuracy: {model.score(X,y)} ' )
    #retunrs all scores in a datframe.

    df['Scores'] = scores
    df['Model'] = [name for name, model in pipelines]
    df.set_index('Model')
    
    return df


# In[ ]:


def get_gap(pipelines,train_scores,test_scores):
    ''' 
    for assesing the gap between train and test model scores
    '''
    output = []
    
    for i , li in enumerate(train_scores):
        dif =  li - test_scores[i]
        output.append(dif)
    result = pd.DataFrame()
    result['Model'] = [name for name , model in pipelines]
    result['Gap'] = [li for li in output]
    
    return result


# In[11]:


def get_gap(pipelines,train_scores,test_scores):
    ''' 
    for assesing the gap between train and test model scores
    '''
    output = []
    
    for i , li in enumerate(train_scores):
        dif =  li - test_scores[i]
        output.append(dif)
    result = pd.DataFrame()
    result['Model'] = [name for name , model in pipelines]
    result['Gap'] = [li for li in output]
    
    return result


# In[13]:



def best_score(pipelines,X,y):
    """
    returns the name and score of the model with the highest test score
    """
    
    best_accuracy = 0.0
    best_classifier = '' 
    best_pipeline = ''
    
    
    
    for name, model in pipelines:
        if model.score(X,y) > best_accuracy:
            best_accuracy = model.score(X, y)
            best_pipeline = model
            best_classifier = name
            
    print(f' Best Accuracy: {best_accuracy}')
    print(f'Model: {best_classifier}')


# In[16]:


def make_results_df(pipelines,X_train,y_train,X_test,y_test):
    
    '''
    This function scores test and train population for all models in pipelines list
    The Gap is calculated by taking the difference in scores
    A dataframe is returned with train, test , and gap values for each model.
    '''
    
    train_scores = score_pipes(pipelines,X_train,y_train)
    test_scores = score_pipes(pipelines,X_test,y_test)
    output = pd.merge(left = train_scores,right =  test_scores, left_on = 'Model',right_on = 'Model')
    train_score_list = [ score for score in train_scores['Scores']]
    test_score_list = [ score for score in test_scores['Scores']]
    gap_results = get_gap(pipelines,train_score_list,test_score_list)
    output = pd.merge(output, gap_results, left_on='Model', right_on='Model')
    output.set_index('Model')
    return output
    
    


# In[ ]:




