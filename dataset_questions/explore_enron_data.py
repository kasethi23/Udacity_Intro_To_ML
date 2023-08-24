#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import pandas as pd

enron_data = joblib.load(open("/content/ud120-projects/final_project/final_project_dataset.pkl", "rb"))

'''print(len(enron_data))
length=[len(value) for value in enron_data.values()]
print(length)'''

'''count = 0
for person in enron_data:
  if enron_data[person]['poi']==1:
    count+=1
print(count)
print(enron_data["PRENTICE JAMES"]['total_stock_value'])
print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])
print(enron_data["SKILLING JEFFREY K"]['exercised_stock_options'])'''
df = pd.DataFrame(enron_data).T
print(df[df.index.str.contains("SKILLING", ) | df.index.str.contains("LAY") |
         df.index.str.contains("FASTOW")].total_payments)
print("Valid salaries: " + str((df.salary != "NaN").sum()))
print("Valid email_address: " + str((df.email_address != "NaN").sum()))
print("Percentage of Nan total payments: " + str(((df.total_payments == "NaN").sum()/len(df))*100))
print("Number of Nan total payments: " + str(((df[df.poi == 1].total_payments == "NaN").sum())))
