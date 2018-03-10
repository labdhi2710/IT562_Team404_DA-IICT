from sklearn.decomposition import truncated_svd
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.random_projection import sparse_random_matrix
from scipy import sparse
from numpy.random import RandomState
from gensim.test.utils import  common_corpus,common_dictionary
from gensim.models import LsiModel
import random
import gensim
import csv
import numpy as np
import pandas as pd
import time

data = pd.io.parsers.read_csv('small_normalised_ratings.csv', names=['user_id','book_id','rating'], engine='python', delimiter=',')
#print(data)

out=open("small_normalised_ratings.csv","r")
o_data=csv.reader(out)
o_data=[[row[0],row[1],eval(row[2])] for row in o_data]
out.close()
num_of_rows = np.max(data.user_id.values)
num_of_cols = np.max(data.book_id.values)
print("No of Rows:")
print(num_of_rows)
print("No of Cols:")
print(num_of_cols)
#ratings_mat = np.ndarray(shape=(num_of_rows,num_of_cols), dtype=np.uint8)
ratings_mat = np.zeros((num_of_rows, num_of_cols), dtype=np.int8)

for row in o_data:
	position_row = eval(row[0])-1
	position_col = eval(row[1])-1
	value = row[2]
	ratings_mat[position_row][position_col] = value

print("----")
print("ratings_matrix")
print(ratings_mat)

corpus = gensim.matutils.Dense2Corpus(ratings_mat,documents_columns=True)
#print(corpus)
#print(common_corpus)
start=time.clock()
model = LsiModel(corpus, num_topics=2)
print "Time = ", time.clock()-start

  # train model
	
print("--------\n SVD done using Gensim")
print("U matrix")
print(model.projection.u)
print("--------\n S matrix\n")
print(model.projection.s)
print("--------\n VT matrix\n")
V = gensim.matutils.corpus2dense(model[corpus], len(model.projection.s)).T / model.projection.s
print(V)

