# Natural Language Processing

# Importing the libraries

import numpy as np  # import numpy package for arrays and stuff
import matplotlib.pyplot as plt # import matplotlib.pyplot for plotting our result 
import pandas as pd  # import pandas for importing csv files  
# quoting = 3 for ignoring ""
dataset = pd.read_csv('Restaurant_Reviews.tsv' , delimiter = '\t' , quoting = 3)

#cleaning the text
#sub function removes all things that we wanted to 
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
corpus = []    # list for storing cleaned reviews
for i in range (0,1000):
    #step 1 -emove unwanted things other than letters
    review = re.sub ('[^a-zA-Z]',' ', dataset['Review'][i])
    #step 2 - make every letter lowercase
    review = review.lower()
    #step 3 - remove nonsignificant words i.e preposition etc --- we need nltk library
    review = review.split()
    # we need for loop to parse each element of this review list and remove unwanted element|| set is used because it is fast and can be used with large data viz-book ,review etc
    #step 4- steming the word -i.e making 'loved' = 'love'
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #step 5 - joinig back the seperated words of a list into string
    review = ' '.join(review)
    corpus.append(review)  # all cleaned reviews are stored in corpus
    
# creating Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#fitting NAive Bayes classifier model to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predicting the test set results
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test ,y_pred )
print(cm)
