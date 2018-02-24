import spacy
import sys
import json
from spacy.tokens import Token


nlp=spacy.load('en_core_web_sm')

#doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

#doc = nlp(u"I love data science on analytics vidhya")



for i in open("final.txt","rb"):
    d=i.decode('utf8')
    doc=nlp(d)
    for word in doc:
        if word.pos_=="VERB":

            print(word.text, word.lemma_)