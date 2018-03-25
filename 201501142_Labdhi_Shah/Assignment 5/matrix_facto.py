import numpy as np
import surprise  # run 'pip install scikit-surprise' to install surprise
import time
from surprise import Reader
import os
import psutil


class MatrixFacto(surprise.AlgoBase):
    '''A basic rating prediction algorithm based on matrix factorization.'''

    def __init__(self, learning_rate, n_epochs, n_factors):

        self.lr = learning_rate  # learning rate for SGD
        self.n_epochs = n_epochs  # number of iterations of SGD
        self.n_factors = n_factors  # number of factors

    def train(self, trainset):
        '''Learn the vectors p_u and q_i with SGD'''

        print('Fitting data with SGD...')

        # Randomly initialize the user and item factors.
        p = np.random.normal(0, 0.1, (trainset.n_users, self.n_factors))
        q = np.random.normal(0, 0.1, (trainset.n_items, self.n_factors))

        # SGD procedure
        for _ in range(self.n_epochs):
            for u, i, r_ui in trainset.all_ratings():
                err = r_ui - np.dot(p[u], q[i])
                # Update vectors p_u and q_i
                p[u] += self.lr * err * q[i]
                q[i] += self.lr * err * p[u]
                # Note: in the update of q_i, we should actually use the previous (non-updated) value of p_u.
                # In practice it makes almost no difference.

        self.p, self.q = p, q
        self.trainset = trainset

    def estimate(self, u, i):
        '''Return the estmimated rating of user u for item i.'''

        # return scalar product between p_u and q_i if user and item are known,
        # else return the average of all ratings
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            return np.dot(self.p[u], self.q[i])
        else:
            return self.trainset.global_mean


file_path = os.path.expanduser('~/PycharmProjects/aashay/small_ratings.csv')
reader = Reader(line_format='user item rating', sep=',')


data = surprise.Dataset.load_from_file(file_path, reader=reader)

# kf = KFold(n_splits=5,random_state=0,shuffle=False)  # folds will be the same for all algorithms.
data.split(5)  # split data for 5-folds cross validation

start = time.time()
algo = MatrixFacto(learning_rate=.03, n_epochs=10, n_factors=10)
surprise.evaluate(algo, data, measures=['RMSE'])
end = time.time()
print(end - start)

process = psutil.Process(os.getpid())
print(process.memory_info().rss)
