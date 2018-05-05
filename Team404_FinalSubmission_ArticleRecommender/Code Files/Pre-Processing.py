
# coding: utf-8

# Pre-Processing of Articles for Clustering and Content Based


# Libraries Used:

import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import string



dataframes = {
    "articles": pd.read_csv("shared_articles.csv"),
}


uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

# Removing HTML tags

def stripTagsAndUris(x):
    if x:
        # BeautifulSoup on content
        x = str(x)
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""

for df in dataframes.values():
    df["text"] = df["text"].map(stripTagsAndUris)


# Removing Punctuations

def removePunctuation(x):
    # Lowercasing all words
    x = str(x)
    #x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)


for df in dataframes.values():
    df["text"] = df["text"].map(removePunctuation)


# Removing Stopwords Using NLTK Library

import nltk
nltk.download()

stops = set(stopwords.words("english"))
def removeStopwords(x):
    x = str(x)
    # Removing all the stopwords
    filtered_words = [word for word in x.split() if word not in stops]
    return " ".join(filtered_words)


for df in dataframes.values():
    df["text"] = df["text"].map(removeStopwords)

# Cleaning Data

for df in dataframes.values():

    df["text"] = df["text"].str.replace("\\n\\xc2\\xa0",'')
    df["text"] = df["text"].str.replace("\\n\\n",'')
    df["text"] = df["text"].str.replace("\n\n",'')
    df["text"] = df["text"].str.replace("b \n",'')
    df["text"] = df["text"].str.replace("\\n",'')
    df["text"] = df["text"].str.replace("\\",'')


# Stemming and Lemmatization using NLTK

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
 

stemmer = PorterStemmer()

f = lambda x: len(x[4].split()) -1

for df in dataframes.values():
    df["text"] = df["text"].apply(f)
    df["text"] = " ".join([stemmer.stem(word) for word in df["text"].split()])


words_stem = []
 
for word in words:
    words_stem.append(stemmer.stem(word))



# Correcting the Spellings using textblob:


import textblob
from textblob import TextBlob
def spellingCorrection(x):
    x = str(x)
    corrected_word =  str(TextBlob(x).correct())
    return " ".join(corrected_word)

for df in dataframes.values():
    df["body"] = df["body"].map(spellingCorrection)


# Saving the Final Pre-Processed File

for name, df in dataframes.items():
    df.to_csv(name + "_processed1.csv", index=False)

