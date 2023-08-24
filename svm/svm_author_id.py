#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/content/drive/MyDrive/git_projects/ud120-projects/tools")
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
t0 = time()
clf = SVC(kernel='rbf', gamma='auto', C=10000).fit(features_train, labels_train)
print("training time:" + str(round(time() - t0, 3)) + "s")
accuracy = clf.score(features_test, labels_test)
print("accuracy: " + str(accuracy))
preds = clf.predict(features_test)
print(f"10: {preds[10]}")
print(f"26: {preds[26]}")
print(f"50: {preds[50]}")
print(preds.sum())

"""
features_train = features_train[:len(features_train)//100]
labels_train = labels_train[:len(labels_train)//100]
for c in [10, 100, 1000, 10000]:
    t0 = time()
    clf = SVC(kernel='rbf', gamma='auto', C=c).fit(features_train, labels_train)
    print("training time:" + str(round(time()-t0, 3)) + "s")

    #t0 = time()
    #preds = clf.predict(features_test)
    #print("testing time:" + str(round(time()-t0, 3)) + "s")

    accuracy = clf.score(features_test, labels_test)
    print("C = " + str(c) + ", accuracy: " + str(accuracy))
"""

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
