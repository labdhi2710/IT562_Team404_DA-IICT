## Instructions to run the code:

1. Download both the data files from the Dataset folder.
2. Download the given code files in the same folder as the data files.
3. Run the file - ArticleRecoHybrid.py
4. Proceed as per the instructions on the console
    1. Registered user - Given below are a few valid user ids that can be used for logging in as a registered user of our system:
       * -8845298781299428018
       * -1032019229384696495
       * -1130272294246983140
       * 344280948527967603
       * -445337111692715325
       * -8763398617720485024
       * 3609194402293569455
       * 4254153380739593270
       * 344280948527967603
       * 3609194402293569455
    2. Unregistered user - For an unregistered user, recommendations displayed are the most popular items from each of the 10 clusters generated.
    
    
## Description of code files:

#### ArticleRecoHybrid
Contains the classes - ContentBased, Collaborative, Hybrid, Popularity, and the Final Recommender that we use.
This is the main file that processes the datasets, assigns weights to the actions based on timestamp, generates recommendations and records new interactions.
Currently, the evaluation of different models is commented as it is not required for the system from the user side.
Also, the method to show the past interactions of the user is commented as it is only to compare the recommendations generated with the interests of the user for analysis purposes, which we have included in the report.

#### KMeans
A method from this file is called by the ArticleRecoHybrid.py when it generates recommendation for a new user. This method generates clusters of the articles and recommends the most popular article from each cluster (to ensure that all domains are covered and the user can choose an article from the domain of his interest).
_Not to be run directly. Is called from within another method._

#### Pre-processing
This preprocesses the article dataset for applying KMeans clustering.
_Not to be run directly. Is called from within another method._
