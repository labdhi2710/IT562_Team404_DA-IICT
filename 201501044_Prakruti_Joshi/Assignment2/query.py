import csv
from pprint import pprint
from elasticsearch import Elasticsearch

es = Elasticsearch()

# Boosting Query - promote or demote a result, give negative boost
pprint(es.search(index="books", doc_type="book", body={\
    "query": {
        "boosting":{
            "positive": { "match":{"title":"The"} },
            "negative": { "term": {"title":"And"}},
            "negative_boost":0.5,
            }}
}))


# Compound Query - more than one query, matches using boolean combination of queries
# must, should, must_not, filter
pprint(es.search(index="books", doc_type="book", body={\
  "query": {"bool":\
    {"must": { "match":{"title":"The"} },
     "must_not": { "range": {"year-of-publication": {"gte":"2000","lte":"1994"}}}
    }
   }
  }))


# Minimum Should Match - the minimum no of optional clauses that should match
pprint(es.search(index="books", doc_type="book", body={\
  "query": {"bool":{\
        "minimum_should_match":1,
        "should": [
            {"match":{"title":"Kill"}},
            {"match":{"title":"Mockingbird"}},
                ],
        "must": {
                "range": {
                    "year-of-publication": {"gte":"2000","lte":"1994"}
                    }
                }
         }
        }
}))
