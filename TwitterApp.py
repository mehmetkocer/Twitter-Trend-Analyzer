#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:33:16 2021

@author: mehmet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tweepy
import json
import re
import nltk
from nltk import word_tokenize, FreqDist, bigrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from wordcloud import WordCloud
from collections import Counter 
import preprocessor.api as p
import streamlit as st
import time

# API Keys and Tokens 
consumer_key =""
consumer_secret = ""
access_token = ""
access_token_secret = ""

# Authorization and Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#get trends
US_WOE_ID = 23424977
trends = api.trends_place(US_WOE_ID)[0]['trends']
trends = trends[:10]

#create trend name and query lists
trends_name = []
for t in trends:
     trends_name.append(t["name"])  

st.set_page_config(layout='wide')
col1 = st.sidebar

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Twitter trends analyzer")
st.markdown("""
            This app retrieves top **10** trends from the **Twitter** and provide some data analysis about trends!
            
            """)
expander_bar = st.beta_expander("Learn More")
expander_bar.markdown("""
                      * **Python libraries:** tweepy, pandas, streamlit, numpy, matplotlib, nltk, re, wordcloud, preprocessor
                      * **Creator:** [Mehmet Köçer](https://github.com/Mehmetkocer)
                      """)

def get_query(name):   
     df_trend = pd.DataFrame(trends,columns=['name','url','promoted_content','query','tweet_volume'])
     trend_query = df_trend.loc[df_trend['name'] == name]
     trend_query = trend_query["query"].where(trend_query["name"]==name)
     trend_volume = df_trend.loc[df_trend['name'] == name]
     trend_volume = trend_volume["tweet_volume"].where(trend_volume["name"]==name)
     return trend_query.iloc[0],trend_volume.iloc[0]
selected_name = col1.selectbox('Select a Trend',trends_name)
query,tweet_volume= get_query(selected_name)

@st.cache
def get_data(query):
     search_results = api.search(query + " AND -filter:retweets" , count=100,tweet_mode='extended')
     time.sleep(2)
     tweets_list = []
     for result in search_results:
          tweets_list.append({'text': str(result.full_text),'favorite_count': int(result.favorite_count),'retweet_count': int(result.retweet_count),
                             'source': result.source,'user_follower_counts': result.user.followers_count,
                             'user_friends_count': result.user.friends_count,'user_statuses_count': result.user.statuses_count,
                             'user_favourites_count': result.user.favourites_count,'user_verified': result.user.verified})
     tweet_df = pd.DataFrame(tweets_list, columns = 
                                          ['text', 'favorite_count', 
                                          'retweet_count', 'source', 
                                          'user_follower_counts','user_friends_count','user_statuses_count', 'user_favourites_count', 
                                           'user_verified'])
     tweet_df.to_csv('output.csv',index=False)

get_data(query)

df = pd.read_csv('output.csv')

count = 0
for i in df['text']:
    df["text"].loc[count] = p.clean(i)
    count += 1

data = df['text'].astype(str).str.replace('\d+', '')
lower_text = data.str.lower()

def preprocess_data(data):
    #Removes Numbers
    data = data.astype(str).str.replace('\d+', '')
    lower_text = data.str.lower()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    w_tokenizer =  TweetTokenizer()
 
    def lemmatize_text(text):
        return [(lemmatizer.lemmatize(w)) for w in w_tokenizer.tokenize((text))] 
    def remove_punctuation(words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', (word))
            if new_word != '':
                new_words.append(new_word)
        return new_words
    words = lower_text.apply(lemmatize_text)
    words = words.apply(remove_punctuation)
    return pd.DataFrame(words)

pre_tweets = preprocess_data(df['text'])
df['text'] = pre_tweets

stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: [item for item in x if item not in stop_words])

t_string = ""
for text in df['text']:
    for word in text:
        t_string= t_string + " " + word 
        
t_string = t_string.split(" ")
dist = FreqDist(t_string)
words = dist.keys()

col2,col3 = st.beta_columns((3,2))
#Col 2
col2.subheader('More statistic about the hashtag:')
col2.write("* Number of total tweets: "+ str(int(tweet_volume)))
col2.write("* Avarege length of the tweets: "+ str(int(len(t_string)/df.shape[0]))+" words.")
col2.write('* The average **like** for ' + selected_name + ': '+ str(df['favorite_count'].mean()))
col2.write('* The average **retweet** for ' + selected_name + ': '+ str(df['retweet_count'].mean()))
col2.write('* The average **followers** of users that tweet about ' + selected_name + ': '+ str(int(df['user_follower_counts'].mean())))
col2.write('* The average **friends** (users who follow each others) of users that tweet about ' + selected_name + ': '+ str(int(df['user_friends_count'].mean())))
col2.write('* The average of **total tweets** of users that tweet about ' + selected_name + ': '+ str(int(df['user_statuses_count'].mean())))
col2.write('* The average **favourites** of users that tweet about ' + selected_name + ': '+ str(int(df['user_favourites_count'].mean())))
verified = df['user_verified'].value_counts()
#verified.plot(kind = 'barh',title = 'Is the user verified?',mark_right = True,color=['#17becf', '#d62728'],width=100)
col2.subheader('Percentages of verified users:')
col2.dataframe(verified)
#Col 3
col3.write("")

Counter = Counter(t_string)
most_occur = Counter.most_common(10) 
col3.subheader('Wordcloud of the most frequent words')
wc = WordCloud(width=800, height=400, max_words=50).generate_from_frequencies(dist)
plt.figure(figsize=(12,10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
col3.pyplot()
col2.subheader('Most frequent words')
col2.dataframe(most_occur)
col3.subheader('The most used platforms')
device = df['source'].value_counts()
#device.plot(kind = 'barh',title = 'Which platforms the users used?',mark_right = True,color=['#15616D', '#F17105','#AE8CA3','#A2ABB5','#D0CE7C','#E9B872','#8332AC','#ACE894'],width=100)                                                                                      
col3.bar_chart(device)



