import os
import joblib
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.abspath("/content/drive/MyDrive/git_projects/ud120-projects/tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("/content/drive/MyDrive/git_projects/ud120-projects/final_project/final_project_dataset.pkl", "rb") )
data_dict.pop('TOTAL',None)
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
df = pd.Dataframe(data_dict).T
df['exercised_stock_options']=df['exercised_stock_options'].astype('float') 
df['exercised_stock_options']=df['exercised_stock_options'].astype('float')
max_e =df['exercised_stock_options'].max()
min_e =df['exercised_stock_options'].min()
max_s=df['salary'].max()
min_s=df['salary'].min()
print(max_e,min_e,max_s,min_s)
### your code below
print(data[:,0].max())
print(data[:,1].max())

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("Salary")
plt.ylabel("Bonus")
