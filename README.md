**Sentiment Based Product Recommendation System**

This is ML based application that uses Natural Language Processing (NLP) to analyze customer reviews and determine their sentiment (positive or negative). Based on this sentiment, the system recommends products that align with user preferences and opinions expressed in textual reviews.

Unlike traditional recommendation systems that rely purely on numerical ratings or purchase history, this system goes a step further by understanding the emotional context behind user feedback, making recommendations more personalized and relevant.

It combines Natural Language Processing (NLP) with traditional recommendation logic to enhance product suggestions using user review sentiment. The system preprocesses textual review data using tokenization, stopword removal, and vectorization (e.g., TF-IDF), then classifies sentiment using a machine learning model (e.g., Logistic Regression or Naive Bayes or Random Forest). Sentiment polarity is used to filter or rank product recommendations based on user preferences.

The application is deployed as a Flask web app, with the trained model serialized using pickle for real-time inference. It is fully compatible with cloud deployment platforms like Heroku, enabling scalable, interactive user experiences.

**The sentiment based product recommendation system with Flask app is hosted on Heroku :**

**https://product-recommendation-app-0944ff2ab516.herokuapp.com/**

**Project Structure**

├── app.py                          # Flask web application

├── model.py                        # Machine learning model code

├── templates/                      # HTML templates for the frontend

├── pickle/                         # Serialized model/data files

├── sample30.csv                    # Review dataset

├── requirements.txt                # Python dependencies

├── runtime.txt                     # Python runtime version (for deployment)

├── Procfile                        # Deployment instructions (Heroku)

├── README.md                       # Readme file

├── Sentiment_Based_Product_Recommendation_... # Jupyter notebook (EDA, Preprocessing, Text processing, buiding different models and evaluation, building collaorative filtering models and evaluation, combining sentiments and collaborative filtering techniques, fine tuning the recommendation system)

**Features**

Analyzes sentiment from product reviews using classification model.

Recommends products based on collaborative filtering (using similarity matrix) 

Combining the sentiment and collaborative filtering recommendationdations

Simple and responsive UI using Flask API.

Deployment using Heroku.

**Installation**
https://github.com/PraseethaSingh/Sentiment_based_Product_Recommendation_System

**Dataset**

sample30.csv: Contains review text data used for training/testing the sentiment analysis model.

**Machine Learning Model**

-Built using NLP techniques (e.g., TF-IDF, sentiment scoring).

-Data Serialized and saved in the pickle/ directory.

-Loaded dynamically in the Flask app for real-time predictions.

**Deployment**

The sentiment based product recommendation system with Flask app is hosted on Heroku :
 
 https://product-recommendation-app-0944ff2ab516.herokuapp.com/

**Tools Required**

Python 3.8+
Flask
Pandas, NumPy
Scikit-learn
Jupyter (for notebook usage)

**Author**

Praseetha Kumarsingh

GitHub : https://github.com/PraseethaSingh/Sentiment_based_Product_Recommendation_System



 
