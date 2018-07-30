import pandas as pd
import re
import nltk
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import xmltodict as xml
import random

training_set_file = "/home/hodaya/Documents/INTERNSHIP/LMRNoteRandom.xml"
testing_set_file = "/media/hodaya/DISK_IMG/RPDR/RPDR/LMRNote.xml"


# convert xml to dictionary using xmltodict
with open(training_set_file) as train:
    traning_set_dict = xml.parse(train.read())
with open(testing_set_file) as test:
    testing_set_dict = xml.parse(test.read())

# preprocess the data (stripping special characters and lowercasing)
def preprocessing(data):
    preprocessed = []
    for i, val in enumerate(data['dataroot']['LMRNote']): # i is report index
        empi, epic_pmrn  = val['EMPI'], val['EPIC_PMRN']
        date, record_id, stat = val['LMRNote_Date'], val['Record_Id'], val['Status']
        cod, inst = val['COD'], val['Institution']
        subj, text = val['Subject'], val['Comments']
        text = re.sub('\W+', ' ', text).lower().strip() # taking out special characters and lowercasing
        preprocessed.append((empi, epic_pmrn, date, record_id, stat, cod, inst, subj, text))
    return preprocessed



###########################################################################################

training_set = preprocessing(traning_set_dict) 
testing_set = preprocessing(testing_set_dict)
train_setComm = [t[8] for t in training_set] # access the 9th tuple a.k.a. the reports/comments with the emails
test_setComm = [t[8] for t in testing_set]

# word has to appear at least 5 times to be in vocab
min_df = 5
max_features = 1000
countVec = CountVectorizer(min_df = min_df, max_features = max_features)
# learn the vocabulary from train set (it will include names, date, things we want deidentified! - and other random repeated words)
# transforming to matrix of bag-of-word vectors
''' countVectorizer takes the reports and maps to:
                        vocab, the number of times it appears
                     ########################################
list of train reports#
                     #
'''
trainSet = countVec.fit_transform(train_setComm) # fit learns the vectorizer (i.e builds vocab etc), transform applies it
testSet = countVec.fit_transform(test_setComm)

# classification (logistic regression)
# map x to y: x is the deidentified reports in the training set
# y are the emails in the xml w/o deidentification
lreg = LogisticRegression()
fitted = lreg.fit(trainSet, testSet) # #2 is my target vector relative to trainSet

# what is the accuracy of the logistic regression model?
score = lreg.score(trainSet, testSet)
print('Logistic Regression test accuracy: %.3f' % score)
