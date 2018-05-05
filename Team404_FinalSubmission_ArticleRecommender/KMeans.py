from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import Popularity
    
def k_means_clustering(no_of_clusters, item_popularity_df, articles_df):

    x_df = pd.read_csv('shared_articles.csv')
    x = x_df['text']
    model = KMeans(n_clusters=no_of_clusters, init='k-means++', max_iter=100, n_init=1)
    #model.fit(x)
 
    print("\n")

    popularityModel = Popularity.PopularityRecommender(item_popularity_df, articles_df)
    list_of_popular_items_clusterwise = popularityModel.recommend_items(user_id = 1, topn = no_of_clusters, verbose = True)
    return list_of_popular_items_clusterwise