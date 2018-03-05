import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.random_projection import sparse_random_matrix
from scipy import sparse
from numpy.random import RandomState
import random

# U--> m x r matrix
# Sigma--> r x r matrix 
# r---> n_components
# VT--> r x n matrix
# Which implies X= (m x n) matrix
# X= U Sigma VT

data = pd.io.parsers.read_csv('small_normalised_ratings.csv', names=['user_id','book_id','rating'], engine='python', delimiter=',')
out=open("small_normalised_ratings.csv","r")
o_data=csv.reader(out)
o_data=[[row[0],row[1],eval(row[2])] for row in o_data]
out.close()

num_of_rows = np.max(data.user_id.values)
num_of_cols = np.max(data.book_id.values)
print('number of rows in matrix X: %d '%(num_of_rows))
print('number of columns in matrix X: %d '%(num_of_cols))
ratings_mat = np.zeros((num_of_rows, num_of_cols),dtype=np.int8)

for row in o_data:
	position_row = eval(row[0])-1
	position_col = eval(row[1])-1
	#position = [int(num_of_cols * position_row + position_col)]
	value = row[2]
	ratings_mat[position_row][position_col] = value

print('Matrix X: \n')
print(ratings_mat)
print('\n')
#X.todense()
#print X

svd = TruncatedSVD(n_components=2, n_iter=7, random_state=np.random.randint(0, 2))
svd.fit(ratings_mat)  

U, Sigma, VT = randomized_svd(ratings_mat, n_components=2, n_iter=7,random_state=np.random.randint(0, 2))
print('Matrix U: \n')
print(U)
print('\n')
print('Matrix Sigma: \n')
print(Sigma)
print('\n')
print('Matrix VT: \n')
print(VT)
print('\n')