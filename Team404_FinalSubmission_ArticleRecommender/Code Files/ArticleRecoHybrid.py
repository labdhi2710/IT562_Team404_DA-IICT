import numpy as np
import scipy
import webbrowser
import pandas as pd
import math
import random
import sklearn
import time
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import ModelEval
import Popularity
import ContentBased
import CF
import Hybrid
import Final
import KMeans

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False)                                     .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)]                                .sort_values('recStrength', ascending = False)                                .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, item_ids, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength'])                                     .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df

class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    cb = pd.DataFrame()
    cf = pd.DataFrame()


    def __init__(self, cb_rec_model, cf_rec_model, items_df):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, items_to_ignore=items_to_ignore, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        #Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df,
                                   how = 'inner', 
                                   left_on = 'contentId', 
                                   right_on = 'contentId')
        
        #Computing a hybrid recommendation score based on CF and CB scores
        recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF']
        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrengthHybrid', 'contentId', 'title', 'url', 'lang']]

        cb = cb_recs_df
        cf = cf_recs_df
        #print('list printing : ')
        #print(list(cb['contentId']))
        #print(cf)
        return recommendations_df,cb,cf

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)]                                .sort_values('eventStrength', ascending = False)                                .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics).sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df


class FinalRecommender:
    
    MODEL_NAME = 'Final'
    
    global cb
    global cf
    def __init__(self, hybrid_rec_model, cb_rec_model, cf_rec_model, items_df):
        self.hybrid_rec_model = hybrid_rec_model
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        
        n=topn

        #hybrid_recommender_model = Hybrid.HybridRecommender(cb_rec_model,cf_rec_model,items_df)
        #Getting the top-6 Hybrid-based filtering recommendations
        #print('list printing1 : ')
        #print(list(cb['contentId']))
        hybrid,cb,cf = hybrid_recommender_model.recommend_items(user_id, topn=int(0.6*n), verbose=False)
        #print('list printing : ')
        #print(list(cb['contentId']))
        #print('hrer')
        #global cb
        #global cf
        #print(list(cb['contentId']))
        #Getting the top-10 Content-based filtering recommendations
        #cb = content_based_recommender_model.recommend_items(user_id, topn=int(1.5*n),verbose=False)
        
        #Getting the top-12 Collaborative filtering recommendations
        #cf = cf_recommender_model.recommend_items(user_id, topn=int(1.5*n),verbose=False)
        
        hybrid_list = list(hybrid['contentId'])
        #print('jdhfjehrher')
        cb_list = list(cb['contentId'])
        #print('fhskjhfkjhskjhf')
        cf_list = list(cf['contentId'])
        #print('khfskhfkshkfh')
        lb = len(cb_list)
        lf = len(cf_list)
        #print('len is '+str(lb))
        #print('len is '+str(len(cb_list)))
        #print(min(0.2*n,lb))
        #print(min(0.2*n,lf))
        
        for i in hybrid_list:
            for j in range(min(int(0.2*n),len(cb_list))):
                if str(i)==str(cb_list[j]):
                    del cb_list[j]
        
        for i in hybrid_list:
            for j in range(min(int(0.2*n),len(cf_list))):
                if str(i)==str(cf_list[j]):
                    del cf_list[j]
        
        for i in cb_list:
            for j in range(min(int(0.2*n),len(cf_list))):
                if str(i)==str(cf_list[j]):
                    del cf_list[j]
        
        hybrid_list = hybrid_list[:min(int(0.6*n),len(hybrid_list))]
        cb_list = cb_list[:min(int(0.2*n),len(cb_list))]
        cf_list = cf_list[:min(int(0.2*n),len(cf_list))]

        hybrid_df = pd.DataFrame({'contentId' : hybrid_list})
        cb_df = pd.DataFrame({'contentId' : cb_list})
        cf_df = pd.DataFrame({'contentId' : cf_list})

        hybrid_df_result = pd.merge(hybrid, hybrid_df,on='contentId')[['contentId','recStrengthHybrid']].rename(columns={'recStrengthHybrid':'recStrength'})
        cb_df_result = pd.merge(cb, cb_df,on='contentId')
        cf_df_result = pd.merge(cf, cf_df,on='contentId')

        final_df = hybrid_df_result
        final_df = final_df.append(cf_df_result,ignore_index=True)
        final_df = final_df.append(cb_df_result,ignore_index=True)

        final_df = pd.merge(final_df,articles_df,on='contentId').sort_values('recStrength', ascending = False)[['recStrength', 'contentId', 'title', 'url', 'lang']]
        
        return final_df


articles_df = pd.read_csv('shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED'][['timestamp','eventType','contentId','authorPersonId','contentType','url','title','text','lang']]
#print('# articles: %d' % len(articles_df))
#articles_df.head(5)


# In[3]:

interactions_df = pd.read_csv('users_interactions.csv')
interactions_df = interactions_df[['timestamp','eventType','contentId','personId']]
#interactions_df.head(10)


# In[4]:

comments_only = interactions_df.loc[interactions_df['eventType']=='COMMENT CREATED'].set_index(['personId', 'contentId'])
likes_only = interactions_df.loc[interactions_df['eventType']=='LIKE'].set_index(['personId', 'contentId'])
#print('# comment interactions: %d' % len(comments_only))
#print('# like interactions: %d' % len(likes_only))
comments_with_likes = comments_only.merge(likes_only, 
               how = 'inner',
               left_index=True, right_index=True)[['timestamp_x','eventType_x']].rename(columns={'timestamp_x': 'timestamp', 'eventType_x': 'eventType'})
#comments_with_likes.head(10)
#print('# comments with likes: %d' % len(comments_with_likes))
#interactions_df = interactions_df.merge()


# In[5]:

event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,
   'DISLIKE': -2.5
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])


# In[6]:


mn = float(min(list(interactions_df['timestamp'])))
mx = float(max(list(interactions_df['timestamp'])))
rng = float(1.0*(mx-mn))
val = float(rng*1.0/12.0)

def f(x):
  x = int(x)
  seg_num = int(math.ceil((1.0*(x-mn))/val*1.0))
  tstrength = 1.0 - float(1.0*(seg_num-1)*0.08)

  return tstrength

def g(x):
  return float(x)


interactions_df['timeStrength'] = interactions_df['timestamp'].apply(lambda x: f(x))
interactions_df['eventStrength'] = interactions_df['timeStrength'].apply(lambda x: g(x)) * interactions_df['eventStrength'].apply(lambda x: g(x))
#interactions_df.head(10)


# In[7]:


users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
#print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
#print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))


# In[8]:

#print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')
#print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))
#interactions_from_selected_users_df.head(10)


# In[9]:


'''
r = interactions_from_selected_users_df['timestamp']
splitValue = r.quantile(0.2, interpolation='nearest')
#interactions_before_date_df, interactions_after_date_df = [x for _, x in interactions_from_selected_users_df.groupby(interactions_from_selected_users_df['timestamp'] < splitValue)]

interactions_train_df, interactions_test_df = [x for _, x in interactions_from_selected_users_df.groupby(interactions_from_selected_users_df['timestamp'] < splitValue)]

print('# of before date interactions: %d' % len(interactions_before_date_df))
print('# of after date interactions: %d' % len(interactions_after_date_df))
'''


# In[10]:

def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df.groupby(['personId', 'contentId'])['eventStrength'].sum()                     .apply(smooth_user_preference).reset_index()
'''
interactions_train_df = interactions_before_date_df.groupby(['personId', 'contentId'])['eventStrength'].sum()                     .apply(smooth_user_preference).reset_index()
interactions_test_df = interactions_after_date_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
'''
#print('# of unique user/item interactions: %d' % len(interactions_full_df))
#interactions_full_df.head(10)


# In[11]:

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df, stratify=interactions_full_df['personId'], test_size=0.20, random_state=42)
#print('# interactions on Train set: %d' % len(interactions_train_df))
#print('# interactions on Test set: %d' % len(interactions_test_df))


# In[12]:


#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')


# In[13]:


def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


# In[14]:


#Top-N accuracy metrics consts
#EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100
    
model_evaluator = ModelEvaluator()  


# In[15]:

#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)


# In[16]:
   
popularity_model = Popularity.PopularityRecommender(item_popularity_df, articles_df)


# In[17]:

#print('Evaluating Popularity recommendation model...')
#pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
#print('\nGlobal metrics:\n%s' % pop_global_metrics)
#pop_detailed_results_df.head(10)


# In[18]:

#nltk.download('stopwords')


# In[19]:

#CONTENT BASED

#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()
#print(tfidf_matrix)


# In[20]:

def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['contentId'])
    
    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
    #Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_full_df[interactions_full_df['contentId']                                                    .isin(articles_df['contentId'])].set_index('personId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles

# In[21]:

user_profiles = build_users_profiles()
len(user_profiles)


# In[22]:


myprofile = user_profiles[-1479311724257856983]
#print(myprofile.shape)
pd.DataFrame(sorted(zip(tfidf_feature_names, 
                        user_profiles[-1479311724257856983].flatten().tolist()), key=lambda x: -x[1])[:20],
             columns=['token', 'relevance'])


# In[23]:
    
content_based_recommender_model = ContentBasedRecommender(item_ids, articles_df)


# In[24]:


'''
print('Evaluating Content-Based Filtering model...')
cb_global_metrics, cb_detailed_results_df = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)
'''


# In[25]:


#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength').fillna(0)

users_items_pivot_matrix_df.head(10)


# In[26]:


users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
users_items_pivot_matrix[:10]


# In[27]:


users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]


# In[28]:


#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)


# In[29]:

#U.shape


# In[30]:

#Vt.shape


# In[31]:

sigma = np.diag(sigma)
#sigma.shape


# In[32]:

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
#all_user_predicted_ratings


# In[33]:

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
#cf_preds_df.head(10)


# In[34]:

#len(cf_preds_df.columns)


# In[35]:
    
cf_recommender_model = CFRecommender(cf_preds_df, articles_df)


# In[36]:


'''
print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
cf_detailed_results_df.head(10)
'''


# In[37]:

#cb = pd.DataFrame()
#cf = pd.DataFrame()
  
hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df)


# In[38]:


'''
print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df.head(10)
'''


# In[39]:
    
final_recommender_model = FinalRecommender(hybrid_recommender_model, content_based_recommender_model, cf_recommender_model, articles_df)


# In[40]:


'''
print('Evaluating Final model...')
final_global_metrics, final_detailed_results_df = model_evaluator.evaluate_model(final_recommender_model)
print('\nGlobal metrics:\n%s' % final_global_metrics)
final_detailed_results_df.head(10)
'''


# In[56]:


#global_metrics_df = pd.DataFrame([pop_global_metrics, cf_global_metrics, cb_global_metrics, final_global_metrics]) \
                        #.set_index('modelName')
#global_metrics_df
print('\n************ Welcome to the 404\'s exclusive Article Recommender System ************ \n\n\n')

print('Are you a registered User ? [Y/N]: ')
ch = raw_input('')
out = False
check = 0

if ch=="Y" or ch=="y":
  while True:
    print('\nPlease Recall your userId and enter here: ')
    input_id = int(raw_input(''))
    x = 0
    for i in interactions_train_df['personId']:
      if int(i)==input_id:
        x=1
        print('\nUser FOUND!\n')
        break


    while True:

      if x==1:
        print('Recommended Articles : \n')
        #Call hybrid model here
        if check==0:
          final_list = final_recommender_model.recommend_items(input_id,topn=10, verbose=True)
        
        print(final_list)
        print('\n\nPlease Enter the Article number you want to Read:[0-9] ')
        num = int(raw_input(''))
        print('\n\nOkay! Opening Article.....\n\n')
        article = list(final_list['url'])[num]
        cid = list(final_list['contentId'])[num]
        url = str(article)
        webbrowser.open(url)
        tm = int(time.time())
        new_interaction = pd.Series([tm,'VIEW',cid,input_id])
        interactions_df.append(new_interaction,ignore_index=True)
        print('\n\nInteraction Recorded!!\n\n')
        print('Do you want to do any of the following Operation on the Article you just Read:\n1.Like\n2.Comment\n3.Bookmark\n4.Follow\n5.Dislike\n6.NOTA\n\n')
        op_ch = int(raw_input(''))

        if op_ch!=6:
          op = list(['Like','Comment','Bookmark','Follow','Dislike'])
          #print(op)
          tm = int(time.time())
          #print tm
          new_interaction = pd.Series([tm,op[op_ch-1],cid,input_id])
          interactions_df.append(new_interaction,ignore_index=True)
          #print('len is '+(str(len(interactions_df))))
          print('\n\nInteraction Recorded!!!\n\n')

        print('\n\nWhat you would like to do now: \n0.EXIT\n1.Keep reading the above Recommended Articles\n\n')

        article_ch = int(raw_input(''))

        if article_ch == 0:
          out = True
          break

        else:
          check = 1
          #out = True

      if out==True:
        break

    if out == True:
      break
  else:
    print('\n\nUserId Not found/Incorrect UserId\n\n')

else:
  most_popular_among_cluster_items = KMeans.k_means_clustering(10, item_popularity_df, articles_df)
  print(most_popular_among_cluster_items)


# In[ ]:

'''
get_ipython().magic(u'matplotlib inline')
ax = global_metrics_df.transpose().plot(kind='bar', figsize=(15,8))
for p in ax.patches:
    ax.annotate("%.3f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

'''
# In[ ]:

'''
def inspect_interactions(person_id, test_set=True):
    if test_set:
        interactions_df = interactions_test_indexed_df
    else:
        interactions_df = interactions_train_indexed_df
    return interactions_df.loc[person_id].merge(articles_df, how = 'left', 
                                                      left_on = 'contentId', 
                                                      right_on = 'contentId') \
                          .sort_values('eventStrength', ascending = False)[['eventStrength', 
                                                                          'contentId',
                                                                          'title', 'url', 'lang']]
'''

# In[ ]:


#inspect_interactions(-1479311724257856983, test_set=False).head(20)