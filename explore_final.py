
import sys
import pickle
import os
from feature_format import featureFormat
import pandas as pd
import pprint
#print my_dataset['BHATNAGAR SANJAY']

import os
path = os.path.join(os.getcwd(), "enron_data")
poi = 0
count = 0
key_error = 0
key_error_list = []
feature_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                     'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                     'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'fraction_from_poi']
#includeds newly created features.

total_features = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                     'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                     'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
#total features minus email


def poi_non_poi_mean(enron_data,total_features):
    #uses an pandas array after feature_format is used
    byPoi = enron_data.groupby('poi')
    for feature in total_features[1:]:
        print byPoi[feature].mean()

def remove_non_employees(enron_data):
    enron_only = { person : v for person ,v in enron_data.iteritems() if enron_data[person]['to_messages'] != "NaN"}
    return enron_only



def check_NaN(enron_data,total_features):
    count = 0
    key_error = 0
    key_error_list = []
    print "total data points :",len(enron_data)

    for feature in total_features:
        for person in enron_data:
            try:
                if enron_data[person][feature] == 'NaN':
                    count += 1
            except KeyError:
                key_error += 1
                key_error_list.append((person, feature))
        print feature,"NaN count: ", count
        count = 0
        key_error = 0
    print key_error_list




def main():
    with open("final_project_dataset.pkl", "r") as data_file:
        my_dataset = pickle.load(data_file)
    #print my_dataset['BHATNAGAR SANJAY']
    my_dataset.pop('BHATNAGAR SANJAY')
    my_dataset.pop('TOTAL')
    check_NaN(my_dataset,total_features)
    enron_data = featureFormat(my_dataset, total_features, sort_keys = True)
    enron_data = pd.DataFrame(enron_data)
    print enron_data.head()
    enron_data.columns = total_features
    enron_data.to_csv(path)
    print enron_data['poi'].sum()
    print len(enron_data['poi'])
    poi_non_poi_mean(enron_data, total_features)

'''
    count = 0
    for x in my_dataset:
        print my_dataset[x]
        count +=1
        if count == 5:
            break
'''


if __name__ == '__main__':
    main()

#[('BHATNAGAR SANJAY', 'total_stock_value')]  this person has a missing key

'''
total data points : 143
salary NaN count:  51
deferral_payments NaN count:  106
total_payments NaN count:  21

loan_advances NaN count:  141

bonus NaN count:  64

restricted_stock_deferred NaN count:  126

deferred_income NaN count:  96
total_stock_value NaN count:  19
expenses NaN count:  51
exercised_stock_options NaN count:  44
other NaN count:  53
long_term_incentive NaN count:  80
restricted_stock NaN count:  36

director_fees NaN count:  127

to_messages NaN count:  59
email_address NaN count:  34
from_poi_to_this_person NaN count:  59
from_messages NaN count:  59
from_this_person_to_poi NaN count:  59
shared_receipt_with_poi NaN count:  59
[('BHATNAGAR SANJAY', 'total_stock_value')]
'''

'''
poi
0.0    33.428571
1.0    72.375000
Name: from_poi_to_this_person, dtype: float64

poi
0.0    20.730159
1.0    55.500000
Name: from_this_person_to_poi, dtype: float64
poi
0.0     604.873016
1.0    1281.812500
Name: shared_receipt_with_poi, dtype: float64
'''

'''
0.0    19047.619048
1.0        0.000000
Name: loan_advances, dtype: float64
0.0    83906.174603
1.0        0.000000
Name: restricted_stock_deferred, dtype: float64
0.0    11406.079365
1.0        0.000000
Name: director_fees, dtype: float64
'''