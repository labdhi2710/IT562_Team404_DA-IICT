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