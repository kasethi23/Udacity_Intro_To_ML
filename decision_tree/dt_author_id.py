#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("/content/drive/MyDrive/git_projects/ud120-projects/tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
# print number of features
print(len(features_train[0]))
t0 = time()
clf =DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
print("training time: " + str(round(time()-t0, 3)) + "s") 

accuracy = clf.score(features_test, labels_test)
print(f'Accuracy: {accuracy}')
