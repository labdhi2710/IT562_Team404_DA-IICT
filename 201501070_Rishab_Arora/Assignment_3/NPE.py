import spacy
import sys
import json
from spacy.tokens import Token

'''
This is a python code for Noun Phrase Extraction using spacy.
We have implemented Noun Phrase Extraction on the long description of the Book from the dataset of books. 
We had to first extract the text file from the json object.
'''


nlp=spacy.load('en_core_web_sm')

#doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

#doc = nlp(u"I love data science on analytics vidhya")

#

for i in open("final.txt","rb"):
    d=i.decode('utf8')
    doc=nlp(d)
    for word in doc:
            print(word.text, word.lemma_)

# Noun-Chunks

for i in open("final.txt","rb"):
    d=i.decode('utf8')
    doc=nlp(d)
    for word in doc.noun_chunks:
        print(word.text)