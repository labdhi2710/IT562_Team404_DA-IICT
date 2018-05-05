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
    
model_evaluator = ModelEval.ModelEvaluator()  


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
    
content_based_recommender_model = ContentBased.ContentBasedRecommender(item_ids, articles_df)


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
    
cf_recommender_model = CF.CFRecommender(cf_preds_df, articles_df)


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
  
hybrid_recommender_model = Hybrid.HybridRecommender(content_based_recommender_model, cf_recommender_model, articles_df)


# In[38]:


'''
print('Evaluating Hybrid model...')
hybrid_global_metrics, hybrid_detailed_results_df = model_evaluator.evaluate_model(hybrid_recommender_model)
print('\nGlobal metrics:\n%s' % hybrid_global_metrics)
hybrid_detailed_results_df.head(10)
'''


# In[39]:
    
final_recommender_model = Final.FinalRecommender(hybrid_recommender_model, content_based_recommender_model, cf_recommender_model, articles_df)


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

        print '\n\nWhat you would like to do now: \n0.EXIT\n1.Keep reading the above Recommended Articles\n\n'
        article_ch = int(raw_input(''))

        if article_ch == 0:
          out = True
          break

        else:
          check = 1
          out = True

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