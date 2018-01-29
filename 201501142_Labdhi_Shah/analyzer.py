import csv
from pprint import pprint
from elasticsearch import Elasticsearch

es = Elasticsearch()

#Specify analyzer when creating index
if not es.indices.exists(index="books"):
    pprint(es.indices.create(index="books", body=\
    {
            "settings":{
              "analysis": {
                  "analyzer": {
                    "book_title_analyzer": {
                      "type": "custom"
                      "tokenizer": "pattern"
                      "filter": "lowercase"
                    }
                  }
            }
            },
            "mappings":{
            "book":{
                "properties":{
                    "isbn":{"type":"keyword"},
                    "book-title":{"type":"text", "analyzer": "book_title_analyzer", "index_options":"positions"},
                    "book-author":{"type":"text"},
                    "year-of-publication":{"type":"integer"}
                    "publisher":{"type":"text"},
                    "image-url-s":{"type":"text"},
                    "image-url-m":{"type":"text"},
                    "image-url-l":{"type":"text"},
                    }
                }
            }
        }
