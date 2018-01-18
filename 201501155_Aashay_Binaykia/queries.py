import csv
from pprint import pprint
from elasticsearch import Elasticsearch

es = Elasticsearch()

# Prints document from known id value
res1=es.get(index="books", doc_type="book", id=42)
print(res1['_source'])

# Returns all documents
pprint(es.search(index="books", doc_type="book", body={"query":{"match_all":{}}}))

# Prints specific document via keyword
pprint(es.search(index="books", body={"query": {"match": {"any":"data1"}}}))

# Searches by matching exactly : returns all documents with author containing David
pprint(es.search(index="books", doc_type="book", body={"query":{"match":\
{"book-author":"David"}}}))

# Searches by matching phrase : returns documents with title containing the entire phrase
pprint(es.search(index="books", doc_type="book", body={"query":\
{"match_phrase":{"book-title":"Clara Callan"}}}))

# Searches by matching prefix
pprint(es.search(index="books", doc_type="book", body={"query":\
{"match_phrase_prefix":{"book-title":"Goodb"}}}))
