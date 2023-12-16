from statistics import LinearRegression
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')

english_stopwords = stopwords.words('english')
french_stopwords = stopwords.words('french')
italian_stopwords = stopwords.words('italian')
german_stopwords = stopwords.words('german')
arabic_stopwords = stopwords.words('arabic')


testingpercent = 10

datalist = list()
labellist = list()

df = pd.read_csv('./data/en_news_titles.csv')
df = df.sample(frac=1)  # shuffle rows
datalist = [s.strip() for s in list(df['Post Text'])]
labellist = list(df['Subreddit'])

# lemmatization and stop word removal (switch out language_stopwords when preprocessing different languages)
def preprocess(language_stopwords, data):
    lemmatizer = WordNetLemmatizer()
    pp = []
    for i in range(len(data)):
        words = nltk.word_tokenize(data[i])
        words = [lemmatizer.lemmatize(word) for word in words if word not in language_stopwords]
        pp.append(' '.join(words))
    return pp

#get data here, put data in datalist and labels in labellist
#alternatively, if training and testing data is already split can place in  train/test lists to begin with and skip the next bit
#iirc there is also a method from scikitlearn to automatically split data, and even randomizes. can look into it if we want that, but may not be relevant

traindata = list()
trainlabel = list()

testdata = list()
testlabel = list()

for i in range(len(labellist)):
    if i % 100 < 100 - testingpercent:
        traindata.append(datalist[i])
        trainlabel.append(labellist[i])
    else:
        testdata.append(datalist[i])
        testlabel.append(labellist[i])

print('percentage of data used for testing:', (len(testdata) / len(labellist)) * 100)


# # European languages (french and german)
# eu_train = pd.read_json('./data/fr_it_de_train.jsonl', lines=True)
# eu_test = pd.read_json('./data/fr_it_de_test.jsonl', lines=True)

# # french data
# traindata = list(eu_train[eu_train['language'] == 'fr']['comment'])
# trainlabel = list(eu_train[eu_train['language'] == 'fr']['label'])
# testdata = list(eu_test[eu_test['language'] == 'fr']['comment'])
# testlabel = list(eu_test[eu_test['language'] == 'fr']['label'])

# # german data
# traindata = list(eu_train[eu_train['language'] == 'de']['comment'])
# trainlabel = list(eu_train[eu_train['language'] == 'de']['label'])
# testdata = list(eu_test[eu_test['language'] == 'de']['comment'])
# testlabel = list(eu_test[eu_test['language'] == 'de']['label'])

#each model has many settings we can tweak, such as L1/L2 regularization, character vs words, stopwords, n-gram, etc
#probably other models that i havent included
lr = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='log_loss')),
])
lr.fit(traindata, trainlabel)
pred = lr.predict(testdata)
counter = 0
for i in range(len(testlabel)):
    if pred[i] == testlabel[i]:
        counter += 1
print(f"Accuracy for linear regression is {counter / len(testdata)}.")

svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge')),
]) 
svm.fit(traindata, trainlabel)
pred = svm.predict(testdata)
counter = 0
for i in range(len(testlabel)):
    if pred[i] == testlabel[i]:
        counter += 1
print(f"Accuracy for SVM is {counter / len(testdata)}.")

nb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', BernoulliNB()),
])
nb.fit(traindata, trainlabel)
pred = nb.predict(testdata)
counter = 0
for i in range(len(testlabel)):
    if pred[i] == testlabel[i]:
        counter += 1
print(f"Accuracy for NB is {counter / len(testdata)}.")
