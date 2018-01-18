import csv
from elasticsearch import Elasticsearch
import unicodedata
import json

FILE = "fbooks.csv"

MyElasticSearchHost = {
    "host" : "localhost", 
    "port" : 9200
}

INDEX_NAME = 'library'
TYPE_NAME = 'books'
ID_FIELD = 'isbn'

response = open(FILE,"r+")
#output=open("out.csv","w+")
#print(response)

csv_file_object = csv.reader(response)
 
header_row = csv_file_object.next()
#print('header_row_row is %s'%(header_row))
header_row = [item.lower() for item in header_row]

#for item in header_row:
#    print(item)

bulk_data = [] 
for row in csv_file_object:
    mydict = {}
    #row.encode('utf-8')
    for i in range(len(row)):
        #print('row[i] is %s'%(row[i]))
        mydict[header_row[i]] = row[i].decode('utf-8')
    indx_dict = {
        "index": {
        	"_index": INDEX_NAME, 
        	"_type": TYPE_NAME, 
        	"_id": mydict[ID_FIELD]
        }
    }
    bulk_data.append(indx_dict)
    bulk_data.append(mydict)

#for i in bulk_data:
#    print(i)        
# create ES client, create index

es = Elasticsearch(hosts = [MyElasticSearchHost])

if es.indices.exists(INDEX_NAME):
    print("deleting '%s' index..." % (INDEX_NAME))
    res = es.indices.delete(index = INDEX_NAME)
    print(" response: '%s'" % (res))

request_body = {
    "settings" : {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}

print("creating '%s' index..." % (INDEX_NAME))
res = es.indices.create(index = INDEX_NAME, body = request_body)
print(" response: '%s'" % (res))

#spamwriter = csv.writer(output, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
print("bulk indexing...")
res = es.bulk(index = INDEX_NAME, body = bulk_data, refresh = True)
#print(" response: '%s'" % (res))    
print('Document Indexing and bulk indexing successfull!')
choice=1

#DSL QUERIES
while choice:
    print('The console can handle the following queries:(Enter your choice)')
    print('0. Exit\n1. List all books\n2. List all the books with title containing word "Harry Potter"\n3. List all the books with title containing phrase "The:Millennium Trilogy"')
    print('4. List all the books with price in the range of [500,1000]\n5. List all the books whose author is "J.K.Rowling" and price is in the range[500,1000]')
    print('6. List all the books with class "Children fiction"\n')
    choice=int(input('Enter your choice: '))
    if choice==0:
        break

    elif choice==1:
        res = es.search(index = INDEX_NAME, size=10, body={"query": {"match_all": {}}})
        #print(" response: '%s'" % (res))

    elif choice==2:
        res=es.search(index= INDEX_NAME,body={"query":{"match":{"title":"Harry Potter"}}})
        #print(" respnse: '%s'"%(res))

    elif choice==3:
        res=es.search(index=INDEX_NAME,body={"query":{"match_phrase":{"title":"The:Millennium Trilogy"}}})
        #print("response: '%s'"%(res))

    elif choice==4:
        res=es.search(index=INDEX_NAME,body={"query":{"range":{"asp":{"gt":500}}}})
        #print("'response: '%s'"%(res))

    elif choice==5:
        res=es.search(index=INDEX_NAME,body={"query":{"bool":{"must":{"author":"Rowling, J  K"},"must":{"range":{"asp":{"gt":500}}}}}})
        #print("response '%s'"%(res))
        
    elif choice==6:
        res=es.search(index=INDEX_NAME,body={"query":{"bool":{"must":{"match_all":{}},"must":{"match":{"product class":"Children's Fiction"}}}}})
        #print("response '%s'"%(res))

    #json.dumps()
    print('Results: ')
    for hit in res['hits']['hits']:
        print(hit["_source"])