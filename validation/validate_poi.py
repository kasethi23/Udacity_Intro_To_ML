#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("/content/drive/MyDrive/git_projects/ud120-projects/tools"))
from feature_format import featureFormat, targetFeatureSplit


data_dict = joblib.load(open("/content/drive/MyDrive/git_projects/ud120-projects/final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list,sort_keys='/content/drive/MyDrive/git_projects/ud120-projects/tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!  
from sklearn.tree import DecisionTreeClassifier
# unvalidated
clf = DecisionTreeClassifier().fit(features, labels)
print(f'Unvalidated: {clf.score(features, labels)}')

# validated
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=42, test_size=0.3)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

print(f'Validated: {clf.score(x_test, y_test)}')
