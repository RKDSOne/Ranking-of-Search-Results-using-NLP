#### Script to rank the relevance of search results from E-commerce sites ###


#Libraries

import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer




################################################## DATA Preprocessing ###################################################

#Loading data into DataFrame
train = pd.read_csv('train.csv').fillna("")
test = pd.read_csv('test.csv').fillna("")

#Storing important columns in a list
query_list = train['query'].values 
product_title_list = train['product_title'].values

#Dropping the irrelevant ID columns
id_test = test.id.values.astype(int)
train = train.drop("id",axis=1)
test = test.drop("id",axis=1)

#Label
y = train['median_relevance'].values

#Dropping irrelevant columns
train = train.drop(["median_relevance","relevance_variance"],axis=1)
np_y = np.array(y)


#Concatenating the query and the product title to form a single sentence which will be converted to a feature - Tf-idf form
traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

#An alternative to the above functions
temp = train.apply(lambda x: x['query'] + ' ' +  x['product_title'],axis=1)

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
			strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
			ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
			stop_words = 'english')

#Fitting into the tf-idf vectorizer
tfv.fit(traindata)
X =  tfv.transform(traindata) 


X_np = X.toarray()






