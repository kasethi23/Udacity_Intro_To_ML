#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    
    for i in range(len(predictions)):
      errors=(net_worths[i] - predictions[i])**2
      cleaned_data.append((ages[i],net_worths[i],errors))
    cleaned_data = sorted(cleaned_data,key= lambda x : x[2],reverse=True)
    limit=int(len(predictions)*0.9)
    cleaned_data=cleaned_data[:limit]

    
    return cleaned_data
