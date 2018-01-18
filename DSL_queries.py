1. List all books:
	res = es.search(index = INDEX_NAME, size=10, body={"query": {"match_all": {}}})

2. List all the books with title containing word "Harry Potter":
	res=es.search(index= INDEX_NAME,body={"query":{"match":{"title":"Harry Potter"}}})

3. List all the books with title containing phrase "The:Millennium Trilogy":
	res=es.search(index=INDEX_NAME,body={"query":{"match_phrase":{"title":"The:Millennium Trilogy"}}})

4. List all the books with price greater than 500:
	res=es.search(index=INDEX_NAME,body={"query":{"range":{"asp":{"gt":500}}}})

5. List all the books whose author is "J.K.Rowling" and price greater than 500:
	res=es.search(index=INDEX_NAME,body={"query":{"bool":{"must":{"author":"Rowling, J  K"},"must":{"range":{"asp":{"gt":500}}}}}})

6. List all the books with class "Children fiction":
	res=es.search(index=INDEX_NAME,body={"query":{"bool":{"must":{"match_all":{}},"must":{"match":{"product class":"Children's Fiction"}}}}})
