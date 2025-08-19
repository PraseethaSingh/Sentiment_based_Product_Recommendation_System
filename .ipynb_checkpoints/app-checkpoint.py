#!/usr/bin/env python
# coding: utf-8

# In[7]:


from flask import Flask, render_template, request, redirect, url_for, session
import model

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch all usernames from the similarity matrix
    #allUsername = model.recommendation_model.index.tolist()
    allUsername = model.user_similarity_matrix.index.tolist()
    return render_template('index.html', usernameList = allUsername)

@app.route('/recommend', methods = ['POST'])
def recommend():
    username = str(request.form.get('username'))
    
    print('username ', username)
    if not username:
        return redirect(url_for('index'))

    result = model.get_recommendations_new(username)

     # Handle case where username not found (your function returns error string, None)
    if isinstance(result[0], str) and result[1] is None:
        allUsername = model.user_similarity_matrix.index.tolist()
        return render_template('index.html', usernameList=allUsername, error=result[0])
        
    # Otherwise unpack the results
    productNameList, productManufacturerList, posHybridScoreList = result
    productList = zip(productNameList, productManufacturerList, posHybridScoreList)


    #if  posSentimentRateList == None or type(productNameList) == 'str':
    #    allUsername = model.recommendation_model.index.tolist()
    #    return render_template('index.html', usernameList = allUsername, error = productNameList)

   # productList = zip(productNameList, productBrandList, productManufacturerList,posSentimentRateList)

    return render_template('recommendations.html', username = username, productList = productList)


if __name__ == '__main__':
    print('===============Flask App Started=============')
    print('Sentiment Based Product Recommendation System')
    app.run(debug = True)


# In[ ]:




