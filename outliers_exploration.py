import pickle
import sys
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
import numpy as np

#Suprised that Mark Frevert is a non-poi due to his high compensation. I assumed all high paid executives would have some knowledge of corruption at Enron.
#I alos thought Lou Pai would be a POI since
#from watching the Enron movie that is what i thought was implied but he also is not listed as one.
#Going to remove 'BHATNAGAR SANJAY' He is the only data point to have negative restricted stock
#and other data points are the exact same which makes me think there is data corruption such as
#total_payments and Res stock deferred being equal as well as other and
#director_fees being the same value. It makes me think that some of the values have been coppied over.



'''
BHATNAGAR SANJAY
{'salary': 'NaN', 'to_messages': 523, 'deferral_payments': 'NaN', 'total_payments': 15456290, 'exercised_stock_options': 2604490,
'bonus': 'NaN', 'restricted_stock': -2604490, 'shared_receipt_with_poi': 463, 'restricted_stock_deferred': 15456290,
'total_stock_value': 'NaN', 'expenses': 'NaN', 'loan_advances': 'NaN', 'from_messages': 29,
'other': 137864, 'from_this_person_to_poi': 1, 'poi': False, 'director_fees': 137864,
'deferred_income': 'NaN', 'long_term_incentive': 'NaN', 'email_address': 'sanjay.bhatnagar@enron.com', 'from_poi_to_this_person': 0}
'''
#data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 'NaN'

data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

def graph_outliers(data_dict, fin_features):
    #get a basic idea for distribution of financial features for further outlier investigation
    for x in range(len(fin_features)/2):
        features = [fin_features[-1], fin_features[-2]]
        data = featureFormat(data_dict, features)
        for point in data:
            feat1 = point[0]
            feat2 = point[1]
            matplotlib.pyplot.scatter(feat1,feat2)

        matplotlib.pyplot.xlabel(features[0])
        matplotlib.pyplot.ylabel(features[1])
        matplotlib.pyplot.show()
        for x in range(2):
            fin_features.pop()


def check_outliers(data_dict, check_feat, reverse_sort= True):
    for x in range(len(check_feat)):
        feature = [check_feat[-1]]
        check_feat.pop()
        data = featureFormat(data_dict, feature)
        if reverse_sort:
            sortd = np.sort(data, axis = 0)[::-1]
            top_bot = "Top"
        else:
            sortd = np.sort(data, axis = 0)
            top_bot = "Bottom"
        outliers = sortd[0:5][:]
        output = grab_outliers(data_dict, outliers, feature)
        print top_bot, feature[0]
        print output
        print  '-' * 25

def grab_outliers(data_dict, data_points, feature):
    output = {}
    for person in data_dict:
        for x in data_points:
            if x[0] == data_dict[person][feature[0]]:
                output[person] = x[0]
    return output

def pull_persons(data_dict, persons):
    for person in persons:
                print person
                print data_dict[person]



check_feat = ['salary', 'deferral_payments', 'loan_advances', 'restricted_stock_deferred', 'exercised_stock_options', 'other', 'long_term_incentive']
check_neg_feat = ['deferred_income', 'restricted_stock']
fin_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                     'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                     'restricted_stock', 'director_fees']

persons = ['BHATNAGAR SANJAY', 'FREVERT MARK A', 'MARTIN AMANDA K', 'PAI LOU L', 'RICE KENNETH D', 'FASTOW ANDREW S']


graph_outliers(data_dict, fin_features)
check_outliers(data_dict, check_feat)
check_outliers(data_dict, check_neg_feat, reverse_sort=False)
#pull_persons(data_dict, persons)

'''
Top long_term_incentive
{'LAVORATO JOHN J': 2035380.0, 'TOTAL': 48521928.0, 'ECHOLS JOHN B': 2234774.0, 'LAY KENNETH L': 3600000.0, 'MARTIN AMANDA K': 5145434.0}
-------------------------
Top other
{'FREVERT MARK A': 7427621.0, 'TOTAL': 42667589.0, 'LAY KENNETH L': 10359729.0, 'BAXTER JOHN C': 2660303.0, 'MARTIN AMANDA K': 2818454.0}
-------------------------
Top exercised_stock_options
{'RICE KENNETH D': 19794175.0, 'SKILLING JEFFREY K': 19250000.0, 'LAY KENNETH L': 34348384.0, 'TOTAL': 311764000.0, 'HIRKO JOSEPH': 30766064.0}
-------------------------
Top restricted_stock_deferred
{'CHAN RONNIE': -32460.0, 'BELFER ROBERT': 44093.0, 'GATHMANN WILLIAM D': -72419.0, 'JAEDICKE ROBERT': -44093.0, 'BHATNAGAR SANJAY': 15456290.0}
-------------------------
Top loan_advances
{'FREVERT MARK A': 2000000.0, 'TOTAL': 83925000.0, 'LAY KENNETH L': 81525000.0, 'PICKERING MARK R': 400000.0}
-------------------------
Top deferral_payments
{'FREVERT MARK A': 6426990.0, 'TOTAL': 32083396.0, 'HORTON STANLEY C': 3131860.0, 'ALLEN PHILLIP K': 2869717.0, 'HUMPHREY GENE E': 2964506.0}
-------------------------
Top salary
{'FREVERT MARK A': 1060932.0, 'SKILLING JEFFREY K': 1111258.0, 'LAY KENNETH L': 1072321.0, 'PICKERING MARK R': 655037.0, 'TOTAL': 26704229.0}
-------------------------
Bottom restricted_stock
{'CHAN RONNIE': 32460.0, 'GILLIS JOHN': 75838.0, 'JAEDICKE ROBERT': 44093.0, 'PIRO JIM': 47304.0, 'BHATNAGAR SANJAY': -2604490.0}
-------------------------
Bottom deferred_income
{'RICE KENNETH D': -3504386.0, 'FREVERT MARK A': -3367011.0, 'TOTAL': -27992891.0, 'ALLEN PHILLIP K': -3081055.0, 'HANNON KEVIN P': -3117011.0}
-------------------------
'''


'''
bot restricted_stock
 'PIRO JIM': 47304.0, 'BHATNAGAR SANJAY': -2604490.0}
-------------------------
bot deferred_income
{'RICE KENNETH D': -3504386.0, 'FREVERT MARK A': -3367011.0, 'BELDEN TIMOTHY N': -2334434.0, 'ALLEN PHILLIP K': -3081055.0, 'HANNON KEVIN P': -3117011.0}
BHATNAGAR SANJAY
{'salary': 'NaN', 'to_messages': 523, 'deferral_payments': 'NaN', 'total_payments': 15456290, 'exercised_stock_options': 2604490, 'bonus': 'NaN', 'restricted_stock': -2604490, 'shared_receipt_with_poi': 463, 'restricted_stock_deferred': 15456290, 'total_stock_value': 'NaN', 'expenses': 'NaN', 'loan_advances': 'NaN', 'from_messages': 29, 'other': 137864, 'from_this_person_to_poi': 1, 'poi': False, 'director_fees': 137864, 'deferred_income': 'NaN', 'long_term_incentive': 'NaN', 'email_address': 'sanjay.bhatnagar@enron.com', 'from_poi_to_this_person': 0}
'''





