from statistics import LinearRegression
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

testingpercent = 10

datalist = list()
labellist = list()

#get data here, put data in datalist and labels in labellist
#alternatively, if training and testing data is already split can place in  train/test lists to begin with and skip the next bit
#iirc there is also a method from scikitlearn to automatically split data, and even randomizes. can look into it if we want that, but may not be relevant

traindata = list()
trainlabel = list()

testdata = list()
testlabel = list()

for i in range(labellist):
    if i % 100 < 100 - testingpercent:
        traindata.append(datalist[i])
        trainlabel.append(labellist[i])
    else:
        testdata.append(datalist[i])
        testlabel.append(labellist[i])

print('percentage of data used for testing:', (len(testdata) / len(labellist)) * 100)

#each model has many settings we can tweak, such as L1/L2 regularization, character vs words, stopwords, n-gram, etc
#probably other models that i havent included
lr = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='log_loss')),
])
lr.fit(traindata, trainlabel)

svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge')),
]) 
svm.fit(traindata, trainlabel)

nb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
nb.fit(traindata, trainlabel)
