# Before running this file, run file index.py so as to create a mapping/indexing
from elasticsearch import Elasticsearch

MyElasticSearchHost = {
    "host" : "localhost", 
    "port" : 9200
}

INDEX_NAME = 'library'
TYPE_NAME = 'books'
ID_FIELD = 'isbn'
es = Elasticsearch(hosts = [MyElasticSearchHost])
choice=1
i=1

while choice:
    f=open("output.txt","w")
    print('The console can handle the following queries:(Enter your choice)')
    print('0. Exit\n1. Should Query')
    print('2. List all the books with price in the range of [500,1000]\n3. List all the books whose author is "J.K.Rowling" and price is in the range[500,1000]')
    print('4. List all the books with class "Children fiction"\n5. Not Query\n6. Bool query\n')
    choice=int(input('\nEnter your choice: '))
    
    if choice==0:
        break

    elif choice==1:
        res=es.search(index=INDEX_NAME,body={"query":{"bool":{"should":[{"match":{"title":"Harry"}},{"match":{"title":"Girl"}}]}}})

    elif choice==2:
        res=es.search(index=INDEX_NAME,body={"query":{"range":{"asp":{"gt":500}}}})
        #print("'response: '%s'"%(res))

    elif choice==3:
        res=es.search(index=INDEX_NAME,body={"query":{"bool":{"must":{"author":"Rowling, J  K"},"must":{"range":{"asp":{"gt":500}}}}}})
        #print("response '%s'"%(res))
        
    elif choice==4:
        res=es.search(index=INDEX_NAME,body={"query":{"bool":{"must":{"match_all":{}},"must":{"match":{"product class":"Children's Fiction"}}}}})
        #print("response '%s'"%(res))

    elif choice==5:
        res=es.search(index=INDEX_NAME,body={"query":{"bool":{"must_not":{"match":{"publisher group":"Transworld Grp"}}}}})

    elif choice==6:
        res=es.search(index=INDEX_NAME,body={"query":{"bool":{"must":{"match":{"product class":"Children's Fiction"}},"must_not":{"match":{"title":"Harry Potter"}}}}})

    #json.dumps()
    print('Query Executed Successfully!\n')

    print('Query results are in a file "output.txt" in the current directory.\n\n')
    f.write('Query results: %d \n\n'%(i))
    for hit in res['hits']['hits']:
        f.write(str(hit["_source"]))
        f.write('\n\n\n')
    
    print('Do you want to continue?(Y/N): ')
    s=raw_input('')
    if s=="N" or s=="n":
        break