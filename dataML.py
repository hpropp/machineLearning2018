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

training_set_file = r"C:\Users\HE077\Downloads\Pathology (2).xml"
testing_set_file = r"C:\Users\HE077\Downloads\LMRNote (1).xml"

# convert xml to dictionary using xmltodict
with open(training_set_file) as train:
    training_set_dict = xml.parse(train.read())
with open(testing_set_file) as test:
    testing_set_dict = xml.parse(test.read())

# preprocess the data (stripping special characters and lowercasing)
def preprocessEmail(data):
    preprocessed = []
    for i, val in enumerate(data['dataroot']['LMRNote']): # i is report index
        empi, epic_pmrn  = val['EMPI'], val['EPIC_PMRN']
        date, record_id, stat = val['LMRNote_Date'], val['Record_Id'], val['Status']
        cod, inst = val['COD'], val['Institution']
        subj, text = val['Subject'], val['Comments']
        text = re.sub('\W+', ' ', text).lower().strip() # taking out special characters and lowercasing
        preprocessed.append((empi, epic_pmrn, date, record_id, stat, cod, inst, subj, text))
    return preprocessed

def preprocessPath(data):
    preprocessed = []
    for i, val in enumerate(data['dataroot']['Pathology']): # i is report index
        empi, epic_pmrn, mrn_type, mrn,  = val['EMPI'], val['EPIC_PMRN'], val['MRN_Type'], val['MRN']
        num, date, typ = val['Report_Number'], val['Report_Date_Time'], val['Report_Type']
        stat, desc, text = val['Report_Status'], val['Report_Description'], val['Report_Text']
        text = re.sub('\W+', ' ', text).lower().strip() # taking out special characters and lowercasing
        preprocessed.append((empi, epic_pmrn, mrn_type, mrn, num, date, typ, stat, desc, text))
    return preprocessed

###########################################################################################

# preprocess the pathologies and emails
train_set = preprocessPath(training_set_dict)
test_set = preprocessEmail(testing_set_dict)

# what are the lengths of each set?
print("Num Train: {}\n".format(len(train_set)))
print("Num Test: {}\n".format(len(test_set)))

# begin dividing into X and Y
trainText = [t[9] for t in train_set] # access the 10th tuple a.k.a. the reports/comments with the pathology
trainY = [t[1] for t in train_set] # access the 2nd tuple a.k.a. EPIC_PMRN number
testText = [t[8] for t in test_set] # access the 9th tuple a.k.a. the reports/comments with the emails
testY = [t[1] for t in train_set] # access the 2nd tuple a.k.a. EPIC_PMRN number

# countVectorizer - transforming to matrix of bag-of-word vectors
min_df = 5 # word has to appear at least 5 times to be in vocab
max_features = 1000 # build vob only considering top max_features ordered by term frequency across data
countVec = CountVectorizer(min_df = min_df, max_features = max_features)
''' example: takes reports and maps to:
                        vocab, the number of times it appears
                     ########################################
list of train reports#
                     #
'''

# learn the vocab from train and test sets
# get Xs out of your train set
trainX = countVec.fit_transform(trainText) # fit learns the vectorizer (i.e builds vocab etc), transform applies it
testX = countVec.fit_transform(testText)

# understanding the shape/format of countVectorizer
print("Shape of Train X {}\n".format(trainX.shape))
print("Sample of the vocab:\n {}".format(np.random.choice(countVec.get_feature_names(), 20)))

# classification (logistic regression)
# fit X to Y (train_set: reports to # corresponding with patient)
lreg = LogisticRegression()
fitted = lreg.fit(trainX, trainY)

# what is the accuracy of the logistic regression model?
scoreTrain = lreg.score(trainX, trainY)
print('Logistic Regression test accuracy: %.3f:\n' % scoreTrain)
scoreTest = lreg.score(testX, testY) # take X, Y from test and score X to Y
print('Logistic Regression test accuracy: %.3f:\n' % scoreTest)

'''
f = open("C:/Users/HE077/Downloads/testfile.txt", "w")
for line in testY:
    print(line, file = f)
    print
f.close()
'''
