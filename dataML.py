import re
import nltk
import sklearn
import numpy as np
import random
import xmltodict as xml
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

trainingSetFile = r"C:\Users\HE077\Downloads\Pathology (2).xml"
testingSetFile = r"C:\Users\HE077\Downloads\LMRNote (1).xml"

# convert xml to dictionary
with open(trainingSetFile) as train:
    trainDict = xml.parse(train.read())
with open(testingSetFile) as test:
    testDict = xml.parse(test.read())

# preprocess data (strip special characters, lowercasing)
def preprocessEmails(data):
    preprocessed = []
    for i, val in enumerate(data['dataroot']['LMRNote']): # i is report index
        empi, epic_pmrn  = val['EMPI'], val['EPIC_PMRN']
        date, record_id, stat = val['LMRNote_Date'], val['Record_Id'], val['Status']
        cod, inst = val['COD'], val['Institution']
        subj, text = val['Subject'], val['Comments']
        text = re.sub('\W+', ' ', text).lower().strip() # taking out special characters and lowercasing
        preprocessed.append((empi, epic_pmrn, date, record_id, stat, cod, inst, subj, text))
    return preprocessed

def preprocessPathology(data):
    preprocessed = []
    for i, val in enumerate(data['dataroot']['Pathology']): # i is report index
        empi, epic_pmrn, mrn_type, mrn,  = val['EMPI'], val['EPIC_PMRN'], val['MRN_Type'], val['MRN']
        num, date, typ = val['Report_Number'], val['Report_Date_Time'], val['Report_Type']
        stat, desc, text = val['Report_Status'], val['Report_Description'], val['Report_Text']
        text = re.sub('\W+', ' ', text).lower().strip() # taking out special characters and lowercasing
        preprocessed.append((empi, epic_pmrn, mrn_type, mrn, num, date, typ, stat, desc, text))
    return preprocessed

trainSet = preprocessPathology(trainDict)
testSet = preprocessEmails(testDict)

# the set lengths
print("Num Train: {}\n".format(len(trainSet)))
print("Num Test: {}\n".format(len(testSet)))

# start dividing into X and Y
trainText = [t[9] for t in trainSet] # access the 10th tuple a.k.a. the reports/comments with the pathology
trainY = [t[0] for t in trainSet] # access the 2nd tuple a.k.a. EPIC_PMRN number

emails = [t[8] for t in testSet] # access the 9th tuple a.k.a. the reports/comments with the emails
emailsIndexes = [t[0] for t in testSet] # access the 2nd tuple a.k.a. EPIC_PMRN number

# add some of the emails to train data
trainEmails, testText, emailTrainIndex, testY = train_test_split(emails, emailsIndexes, test_size=0.6, random_state=1)
trainText.extend(trainEmails)
trainY.extend(emailTrainIndex)

# check the set length values
print("Num Train X (pathologies & emails): {}\n".format(len(trainText)))
print("Num Train Y (pathologies & emails): {}\n".format(len(trainY)))

print("Num Test X (emails): {}\n".format(len(testText)))
print("Num Test Y (emails): {}\n".format(len(testY)))

# countVectorizer - transforming to matrix of bag-of-word vectors
min_df = 5 # word has to appear at least 5 times to be in vocab
ngram_range = (1,7) # use ngrams to make model more accurate
max_features = 50000 # build vocab only considering top max_features ordered by term frequency across data
stop_words = 'english'
countVec = CountVectorizer(min_df = min_df, ngram_range = ngram_range, max_features = max_features, stop_words = stop_words)

# fit learns the vectorizer (i.e builds vocab from the train set, creates document-term matrix)
countVec.fit(trainText, trainY)

# learn the vocab from the train set, get Xs
trainX = countVec.transform(trainText) # transform applies fit
testX = countVec.transform(testText)

# understanding countVectorizer structure
print("Shape of Train X: {}\n".format(trainX.shape))
print("Shape of Test X: {}\n".format(testX.shape))
#print("Sample of the vocab:\n {}\n".format(np.random.choice(countVec.get_feature_names(), 50)))
# trainY: 44779 KB with EMPI num vs 51668 KB with EPIC_PMRN
# testY: 2450419 KB with EMPI num vs 2827402 KB with EPIC_PMRN

f = open("C:/Users/HE077/Downloads/testfile.txt", "w")
for line in trainY:
    print(line, file = f)
    print
f.close()

f = open("C:/Users/HE077/Downloads/testfile2.txt", "w")
for line in testY:
    print(line, file = f)
    print
f.close()

# classification (logistic regression)
lr = LogisticRegression(C = 1.0)

# fit X to Y (trainSet: reports to # corresponding with patient)
fitted = lr.fit(trainX, trainY)

# what is the accuracy of the logistic regression model?
scoreTrain = lr.score(trainX, trainY)
print('Logistic Regression trainSet test accuracy: %.3f\n' % scoreTrain)
scoreTest = lr.score(testX, testY) # take X, Y from test and score X-->Y
print('Logistic Regression testSet test accuracy: %.3f\n' % scoreTest)

# check the number of classes
classNum = lr.classes_
print('Logistic Regression number of classes: %.3f\n' % len(classNum))

# understanding my results
predictY = lr.predict(testX)
#confusion_matrix = confusion_matrix(testY, predictY)
#confusion_matrix
print(classification_report(testY, predictY))
