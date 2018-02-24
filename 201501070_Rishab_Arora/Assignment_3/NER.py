import spacy
import sys
import json
from spacy.tokens import Token

'''
This is a python code for Named Entity Recognition using Spacy.
The input file is a text file which was obtained by converting a json object of book dataset. 
The NER is applied on the Long Description of the books in the dataset.
The Output of NER is assignment of words to the in-built entities in the spacy.
'''


nlp=spacy.load('en_core_web_sm')

#doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')

#doc = nlp(u"I love data science on analytics vidhya")


for i in open("final.txt","rb"):
    d=i.decode('utf8')
    doc=nlp(d)
    for word in doc:
        labels = set([w.label_ for w in doc.ents])
    for label in labels:
        entities = [e.string for e in doc.ents if label==e.label_]
        entities = list(set(entities))
        print(label, entities)

