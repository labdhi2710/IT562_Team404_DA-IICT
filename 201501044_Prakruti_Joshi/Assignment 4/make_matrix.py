import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.random_projection import sparse_random_matrix
from scipy import sparse
from numpy.random import RandomState
import random
import time


data = pd.io.parsers.read_csv('small_normalised_ratings.csv', names=['user_id','book_id','rating'], engine='python', delimiter=',')
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
ratings_mat = np.zeros((num_of_rows, num_of_cols),dtype=np.int32)

for row in o_data:
	position_row = eval(row[0])-1
	position_col = eval(row[1])-1
	#position = [int(num_of_cols * position_row + position_col)]
	value = row[2]
	#print(repr(position_row) +' '+ repr(position_col) +' '+ repr(value))
	#np.put(ratings_mat, position, value)
	ratings_mat[position_row][position_col] = value
#ratings_mat[data.book_id.values-1, data.user_id.values-1] = data.rating.values

#print(ratings_mat)

print("----")
print("ratings_matrix")

print(ratings_mat)

#X.todense()
#print X
start=time.clock()

svd = TruncatedSVD(n_components=2, n_iter=7, random_state=np.random.randint(0, 2))
svd.fit(ratings_mat)  

# U--> 100 x 5 matrix
# Sigma--> 5 x 5 matrix 
# r---> n_components
# VT--> 5 x 100 matrix
# Which implies X= (100 x 100) matrix

U, Sigma, VT = randomized_svd(ratings_mat, n_components=2, n_iter=7,random_state=np.random.randint(0, 2))
print "Time = ", time.clock()-start
print("--------\n Truncated SVD done using Scikit")
print("U matrix")
print(U)
print("--------\n S matrix\n")
print(Sigma)
print("--------\n V matrix\n")
print(VT)
