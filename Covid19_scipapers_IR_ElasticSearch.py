#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all packages needed
import pandas as pd
import json 
import numpy as np 
import nltk
from nltk import re
from nltk.metrics import *
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


es = Elasticsearch([{'host': 'localhost', 'port':9200}]) #sets up elastic search 
es.ping() #checks status of connection


# In[3]:


df_metadata = pd.read_csv("c:/Users\jacob\Documents\Data\metadata.csv\metadata.csv") #read metadata
df_metadata


# In[4]:


df_meta = df_metadata[df_metadata['sha'].notna()]
df_meta = df_meta[~df_meta['sha'].str.contains(';')]
df_meta_1000 = df_meta[:1000]
df_meta_1000 = df_meta_1000.reset_index(level=None, drop=True, inplace=False, col_level=0, col_fill='')
df_meta_1000
#remove double instances of other json identification, then selet 1000 documents 


# In[5]:


#parsing the json files

directory = 'c:/Users\jacob\Documents\Data\metadata.csv\document_parses\pdf_json/' 

json_files = []
json_contents = []
for filename in df_meta_1000['sha']:
        f = open(directory + filename + '.json')
        json_files.append(filename)
        content = f.read()
        jsons = json.loads(content)
        json_contents.append(jsons)
json_contents


# In[6]:


print(len(json_contents)) #hecking number of jsons parsed


# In[7]:


json_contents[0]['body_text'] #specifically selecting body text


# In[8]:


body_text = []
for n in range(0, 1000):
    body_text.append(json_contents[n]['body_text'])
body_text[0]


# In[9]:


body_text_str= [] #set strng format
for i in body_text:
    body_text_str.append(str(i))

body_text_str[0]


# In[10]:


#cleaning text data using regular expression
body_text_clean = []

for i in body_text_str:
    i = i.replace("\'", "")
    x = re.findall("{text:(.*?)\.,", i)
    body_text_clean.append(x)
    
body_text_clean[0]


# In[11]:


#append it ready for DataFrame
body_text_final = []
for i in body_text_clean:
    body_text_final.append([i])
body_text_final[0]


# In[12]:


#create dataframe 
df_body_text = pd.DataFrame(body_text_final)


results = []
for i in df_body_text[0]:
    i = str(i)
    results.append(i)

df_body_text['body_text'] = results
df_body_text.drop([0], axis = 1, inplace = True)
df_body_text


# In[13]:


df_index = pd.merge(df_meta_1000, df_body_text, left_index=True, right_index=True) #merge dataframes to link body_text


# In[14]:


#set up intal index informative columns 
df_index_final = df_index.loc[:, ['cord_uid','pubmed_id', 'doi', 'url',  'journal', 'license', 'authors', 
                                  'publish_time', 'title', 'abstract', 'body_text']]
df_index_final


# In[15]:


#remove nan values from data
nan_val = df_index_final.isnull().any()
nan_val


# In[16]:


df_index_final.fillna('information not present',inplace=True)


# In[17]:


#converts DataFrame to JSON
df_index_upload = df_index_final.to_dict('records')
df_index_upload[0]


# In[18]:


#settig up generetaor to map to elastic search

def generator(df_index_upload):
    for c, line in enumerate(df_index_upload):
        yield {
        '_index': 'covid_index_base',
        '_type': '_doc',
        '_id':line.get('pubmed_id', None),
        '_source': {
            'cord_uid':line.get('cord_uid',''),
            'doi':line.get('doi',''),
            'url':line.get('url',''),
            'journal':line.get('journal',''),
            'license':line.get('license',''),
            'authors':line.get('authors',''),
            'publish_time':line.get('publish_time',''),
            'title':line.get('title',''),
            'abstract':line.get('abstract',''),
            'body_text':line.get('body_text',None)
        }
            }
#setting up mapping options
settings ={
   "settings": {
    "number_of_shards": 1,
     "similarity": {
       "scripted_tfidf": {
         "type": "scripted",
          "script": {
           "source": "double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; double norm = 1/Math.sqrt(doc.length); return query.boost * tf * idf * norm;"
        }
      }
    }
  },
  "mappings": {
    "dynamic": "true",
    "_source": {
      "enabled": "true"
    },
    "properties": {
      "cord_uid": {
        "type": "text",
        "similarity": "scripted_tfidf"
      },"pubmed_id": {
        "type": "text",
        "similarity": "scripted_tfidf"
       },"doi": {
          "type": "text",
          "similarity": "scripted_tfidf"
       },"url": {
         "type": "text",
          "similarity": "scripted_tfidf"
       },"journal": {
         "type": "text",
          "similarity": "scripted_tfidf"
       },"license": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"authors": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"publish_time": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"abstract": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"body_text": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },
      }
    }
  }

    
#generate generator object        
gen_index = generator(df_index_upload)
next(gen_index)


# In[19]:


#setting up initial index 
Index_Name = 'covid_index_base'
covid_index_pre = es.indices.create(index=Index_Name, ignore=[400,404], body = settings)
upload = helpers.bulk(es, gen_index)


# In[20]:


#nltk sentence tokenising 
 
def SentToke(d):
    nltk_text_sent = []
    for s in d:
        s = str(s)
        s = s.lower()
        s = sent_tokenize(s)
        nltk_text_sent.append(s)
        df_toke_sent = pd.DataFrame({'col':nltk_text_sent})
    return df_toke_sent


# In[21]:


#nltk word tokenising and stemming

def WordToke(d):
    stop_words = stopwords.words('english')

    nltk_text_toke = []
    for w in d:
        w = str(w)
        w = w.lower()
        w = word_tokenize(w)
        w = [w for w in w if not w in stop_words] 
        w = [w.lower() for w in w if w.isalpha()]
        nltk_text_toke.append(w)
    df_toke = pd.DataFrame({'col':nltk_text_toke})
    ps = PorterStemmer()
    df_toke = df_toke.col.apply(lambda x: [ps.stem(word) for word in x])
    df_toke = df_toke.astype(str)
    return df_toke

df_processed_authors = pd.DataFrame(WordToke(df_index_final['authors'])) 
df_processed_title = pd.DataFrame(WordToke(df_index_final['title']))
df_processed_abstract = pd.DataFrame(WordToke(df_index_final['abstract']))
df_processed_bodytext = pd.DataFrame(WordToke(df_index_final['body_text']))


# In[22]:


#combining preprocessed fields 
df_processed_index = pd.DataFrame(df_index_final['pubmed_id'])
df_processed_index = df_processed_index.merge(df_processed_authors, right_index = True, left_index=True)
df_processed_index = df_processed_index.merge(df_processed_title, right_index = True, left_index=True)
df_processed_index = df_processed_index.rename({'col_x':'authors_search', 'col_y':'title_search'}, axis = 1)
df_processed_index = df_processed_index.merge(df_processed_abstract, right_index = True, left_index=True)
df_processed_index = df_processed_index.merge(df_processed_bodytext, right_index = True, left_index=True)
df_processed_index = df_processed_index.rename({'col_x':'abstract_search', 'col_y':'body_text_search'}, axis = 1)
df_processed_index = df_processed_index.drop('pubmed_id', axis = 1)
df_processed_index 


# In[23]:


#merging non-processed fields with pre-processed fields 
df_index_merge = df_index_final.merge(df_processed_index, right_index = True, left_index=True)
df_index_merge 


# In[24]:


#using countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
docs=df_index_merge['body_text_search'].tolist()
cv=CountVectorizer(max_df=0.80, max_features=10000)
word_count_vector=cv.fit_transform(docs)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)


# In[25]:


#using TF-IDF score to find keywords 
def key_matrix(keywords):
    tuples = zip(keywords.col, keywords.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


# In[26]:


#find highest scoring words to be keywords nd dd them to a list
docs_test=df_index_merge['body_text_search'].tolist()
feature_names=cv.get_feature_names()

keyword_lst = []
for doc in docs_test:
    tf_idf_vector=tfidf_transformer.transform(cv.transform([(doc)]))
    sorted_items=key_matrix(tf_idf_vector.tocoo())
    keywords=extract_from_vector(feature_names,sorted_items,10)
    keyword_lst.append(list(keywords))
    
keyword_lst


# In[27]:


#create dataframe from list
df_keywords = pd.DataFrame({'keywords':keyword_lst})
df_keywords['keywords'] = df_keywords['keywords'].astype(str)
df_keywords 


# In[28]:


#merge keywords with rest of the dataframe as final index
df_index_merge_final = df_index_merge.merge(df_keywords, right_index = True, left_index=True)
df_index_merge_final


# In[29]:


#convert to JSON
df_processed_upload = df_index_merge_final.to_dict('records2')
df_processed_upload[0]


# In[30]:


#mapping the second index that will be uploaded
def generator2(js):
    for c, line in enumerate(js):
        yield {
        '_index': 'covid_index_final',
        '_type': '_doc',
        '_id':line.get('pubmed_id', None),
        '_source': {
            'cord_uid':line.get('cord_uid',''),
            'doi':line.get('doi',''),
            'url':line.get('url',''),
            'journal':line.get('journal',''),
            'license':line.get('license',''),
            'authors':line.get('authors',''),
            'publish_time':line.get('publish_time',''),
            'title':line.get('title',''),
            'abstract':line.get('abstract',''),
            'body_text':line.get('body_text',''),
            'authors_search':line.get('authors_search',''),
            'title_search':line.get('title_search',''),
            'abstract_search':line.get('abstract_search',''),
            'body_text_search':line.get('body_text_search',''),
            'keywords':line.get('keywords',None)
        }
            }

settings2 ={
   "settings": {
    "number_of_shards": 1,
     "similarity": {
       "scripted_tfidf": {
         "type": "scripted",
          "script": {
           "source": "double tf = Math.sqrt(doc.freq); double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0; double norm = 1/Math.sqrt(doc.length); return query.boost * tf * idf * norm;"
        }
      }
    }
  },
  "mappings": {
    "dynamic": "true",
    "_source": {
      "enabled": "true"
    },
    "properties": {
      "cord_uid": {
        "type": "text",
        "similarity": "scripted_tfidf"
      },"pubmed_id": {
        "type": "text",
        "similarity": "scripted_tfidf"
       },"doi": {
          "type": "text",
          "similarity": "scripted_tfidf"
       },"url": {
         "type": "text",
          "similarity": "scripted_tfidf"
       },"journal": {
         "type": "text",
          "similarity": "scripted_tfidf"
       },"license": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"authors": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"publish_time": {
        "type": "text",
          "similarity": "scripted_tfidf"
       }, "title": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"abstract": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"body_text": {
        "type": "text",
          "similarity": "scripted_tfidf"
       }, "authors_search": {
        "type": "text",
          "similarity": "scripted_tfidf"
        }, "title_search": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"abstract_search": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },"body_text_search": {
        "type": "text",
          "similarity": "scripted_tfidf"
      }, "keywords": {
        "type": "text",
          "similarity": "scripted_tfidf"
       },
      }
    }
  }

    
        
gen_index2 = generator2(df_processed_upload)
next(gen_index2)


# In[31]:


#final indexing to elastic search
Index_Name2 = 'covid_index_final'
covid_index_processed = es.indices.create(index=Index_Name2, ignore=[400,404], body = settings2)
upload2 = helpers.bulk(es, gen_index2)

