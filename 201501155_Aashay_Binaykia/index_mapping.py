import csv
from pprint import pprint
from elasticsearch import Elasticsearch

es = Elasticsearch()

if not es.indices.exists(index="books"):
    pprint(es.indices.create(index="books", body=\
    {
        "mappings":{
            "book":{
                "properties":{
                    "isbn":{"type":"keyword"},
                    "book-title":{"type":"text","index_options":"positions"},
                    "book-author":{"type":"text"},
                    "year-of-publication":{"type":"integer"}
                    "publisher":{"type":"text"},
                    "image-url-s":{"type":"text"},
                    "image-url-m":{"type":"text"},
                    "image-url-l":{"type":"text"},
                    }
                }
            }
        }))


# Read CSV file and create documents
with open('dataset.csv',encoding="utf-8") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
        pprint(es.index(index="books", doc_type="book", body=\
        {
            "isbn":row[1],
            "book-title":row[2],
            "book-author":row[3],
            "year-of-publication":row[4],
            "publisher":row[5],
            "image-url-s":row[6],
            "image-url-m":row[7],
            "image-url-l":row[8]
}))
