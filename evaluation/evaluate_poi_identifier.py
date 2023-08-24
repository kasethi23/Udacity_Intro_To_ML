#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("/content/drive/MyDrive/git_projects/ud120-projects/tools"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("/content/drive/MyDrive/git_projects/ud120-projects/final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys='/content/drive/MyDrive/git_projects/ud120-projects/tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
poi = [i for i in preds if i==1]

print(f'POIs in test set: {len(poi)}')
print(f'Total number of people: {len(preds)}')
print(f'Validated: {clf.score(x_test, y_test)}')
not_poi = 1 - 4/29
print(f'Accuracy if model predicted only the 0 label: {not_poi}')
def count_tp_fp_fn_tn(preds,actual):
  tp = 0
  fp = 0
  fn = 0
  tn = 0
  for pred,act in zip(preds,actual):
    if bool(pred) and bool(act):
      tp+=1
    elif bool(pred) and not bool(act):
      fp +=1
    elif not bool(pred) and bool(act):
      fn+=1
    elif not bool(pred) and not bool(act):
      tn+=1
  return (tp,fp,fn,tn)
tp = count_tp_fp_fn_tn(preds,y_test)[0]
print(f'True positives: {tp}')
from sklearn.metrics import precision_score,recall_score
print(f'precision_score: {precision_score(y_test,preds)}')
print(f'recall_score: {recall_score(y_test,preds)}')

#Hypothetical true positives 
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
tp_h = count_tp_fp_fn_tn(predictions,true_labels)[0]
print(f'True positives: {tp_h}')
tp_h_1 = count_tp_fp_fn_tn(predictions,true_labels)[3]
print(f'True negatives: {tp_h_1}')
fp = count_tp_fp_fn_tn(predictions,true_labels)[1]
print(f'False positives: {fp}')
fn = count_tp_fp_fn_tn(predictions,true_labels)[2]
print(f'False negatives: {fn}')
print(f'precision_score: {precision_score(true_labels,predictions)}')
print(f'recall_score: {recall_score(true_labels,predictions)}')
