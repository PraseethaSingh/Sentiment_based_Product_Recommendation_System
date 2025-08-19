#!/usr/bin/env python
# coding: utf-8

# In[224]:


import os
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# Get the base directory
base_dir = os.getcwd()

# Get the path for subfolder 'pickle'
pickle_dir = os.path.join(base_dir, "pickle")

#Paths to your files relative to model.py
user_final_rating_path = os.path.join(pickle_dir, "user_final_rating.pkl")
tfidf_vectorizer_path = os.path.join(pickle_dir, "tfidf_vectorizer.pkl")
model_path = os.path.join(pickle_dir, "logistic_regression_model.pkl")
data_path = os.path.join(pickle_dir, "df.pkl")
product_info_path = os.path.join(pickle_dir, "product_id_name_mapped.pkl")


# Load the files
with open(user_final_rating_path, "rb") as f:
    user_similarity_matrix = pickle.load(f)

with open(tfidf_vectorizer_path, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(model_path, "rb") as f:
    sentiment_pred_model_LR = pickle.load(f)

with open(data_path, "rb") as f:
    data_df = pickle.load(f)

with open(product_info_path, "rb") as f:
    product_info = pickle.load(f)


# In[252]:


#Funtion : Get Product recommendations - top 5 products 
def get_recommendations_new(username):
    try:
        recommendations_all = pd.DataFrame(user_similarity_matrix.loc[username]).reset_index()                               

    except KeyError:
        errorMessage = f'Hey Mate! we couldn not find the user "{username}", so we couldn\'t recommend anything \n\
         for "{username}", you can try again by selecting any of the below username to find their recommendations.'
        print(type(errorMessage))
        return errorMessage, None

    # Rename the columns:  'id' → 'product_id' and 'username' -> 'predicted_rating' 
    recommendations_all = recommendations_all.rename(columns={"id": "product_id", username: "predicted_rating"})
    # take top 20 rated products
    recommendations_top20 = recommendations_all.sort_values(by="predicted_rating", ascending=False).head(20)  # take top 20 ratings
    
    # Merge product names
    recommendations_top20 = recommendations_top20.merge(
        product_info,  # select all product details - id, name,brand,manufacturer
        left_on='product_id',
        right_on='id',
        how='left'
    ).drop(columns=['id'])  # drop the duplicate 'id' column

    # Merge all reviews for these top20 products (from all users)
    df_top_reviews = data_df[data_df['id'].isin(recommendations_top20['product_id'])]
    df_top_reviews = df_top_reviews[['id', 'cleaned_reviews_text']]

    #print(df_top_reviews.shape)
    
    # Predict sentiment
    X = tfidf_vectorizer.transform(df_top_reviews['cleaned_reviews_text'].values.astype('U'))
    df_top_reviews['sentiment_predicted'] = sentiment_pred_model_LR.predict(X)
        
    # Compute Positive Sentiment Rate per product

    # calculate total count (= total no of reviews) and sum (= no of positive reviews) 
    psr_df = df_top_reviews.groupby('id')['sentiment_predicted'].agg(
            total_reviews='count',
            positive_reviews='sum'
        ).reset_index()

    # calculate percentage of positive sentiment rate 
    psr_df['positive_sentiment_rate'] = ((psr_df['positive_reviews'] / psr_df['total_reviews']).replace([np.inf, -np.inf], 0).fillna(0) * 100).round(2)

    #print(psr_df.shape)

    #  Merge PSR with recommendations
    recommendations_top20 = recommendations_top20.merge(
            psr_df[['id', 'positive_sentiment_rate']],
            left_on='product_id',
            right_on='id',
            how='left'
        ).drop(columns=['id'])

    ## FINE TUNING THE RECOMMENDATION MODEL

    #sentiment dominates the ranking.
    alpha = 0.3 
    beta = 0.7

    recommendations_top20 = recommendations_top20.rename(columns={"predicted_rating": "cf_score"})
    # Normalize scores
    scaler_cf = MinMaxScaler()
    scaler_psr = MinMaxScaler()
    recommendations_top20['cf_score_norm'] = scaler_cf.fit_transform(recommendations_top20[['cf_score']])
    recommendations_top20['psr_norm'] = scaler_psr.fit_transform(recommendations_top20[['positive_sentiment_rate']].fillna(0))


    # Compute hybrid score
    #beta = 1 - alpha
    recommendations_top20['hybrid_score'] = alpha * recommendations_top20['cf_score_norm'] + beta * recommendations_top20['psr_norm']

    # Round to 2 decimal places
    recommendations_top20['hybrid_score'] = recommendations_top20['hybrid_score'].round(2)

    #recommendations_top20 = recommendations_top20[['product_id','name','hybrid_score']]
    recommendations_top20 = recommendations_top20[['product_id','name','manufacturer','hybrid_score']]
    
    #Sort and pick top-5
    recommendations_top5 = recommendations_top20.sort_values(by='hybrid_score', ascending=False).head(5)

    productNameList = recommendations_top5['name'].tolist()
    productManufacturerList = recommendations_top5['manufacturer'].tolist()
    posHybridScoreList = recommendations_top5['hybrid_score'].tolist()
  
    return productNameList,productManufacturerList,posHybridScoreList
  



