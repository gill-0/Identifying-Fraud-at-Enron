#!/usr/bin/python

import sys
import pickle


from feature_format import featureFormat, targetFeatureSplit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tester import dump_classifier_and_data
from explore_final import check_NaN, poi_non_poi_mean
import email_fraction as ef
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
import pandas as pd
import pprint

def kbest():
  #when run need to modify run_pipe to use select k_best=i
  my_dataset = clean_data(data_dict)
  table = []
  for i in range(1,22):
    clf, my_dataset= run_pipe(tree, tree_param_grid, i)
    precision, recall = test_classifier(clf, my_dataset, feature_list)
    table.append((i, precision, recall))
  with open("kbest_scores.pkl", "w") as file:
        pickle.dump(table, file)
  return table






def plot_kbest():
      with open("kbest_scores.pkl", "r") as file:
        table = pickle.load(file)
      table = pd.DataFrame(table, columns=["Number_of_Features", "Precision", "Recall"])
      table = pd.melt(table, id_vars=["Number_of_Features"], value_vars=["Precision", "Recall"],
                  var_name="Performance_Metric", value_name="Score")
      print table
      sns.pointplot(x="Number_of_Features", y="Score", hue="Performance_Metric", data=table)
      sns.plt.title("Precision and Recall vs Number of Features")
      sns.plt.ylabel("Score")
      sns.plt.xlabel("Kbest Features")
      sns.plt.show()



def test_classifier(clf, my_dataset, feature_list):
    data = featureFormat(my_dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.1, train_size=None, random_state=46)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in sss.split(features,labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    return precision, recall




def run_pipe(classifier, params, i=0):
    #my_dataset = clean_data(data_dict)
    data = featureFormat(my_dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    pipe = Pipeline([
      ('clf', classifier)])

    #,        ('scaler', MinMaxScaler()),
     #   ())
    #()

    gs = GridSearchCV(estimator=pipe,
                    param_grid=params,
                    cv=cross_validator,
                    scoring='f1')
    gs.fit(features,labels)
    clf = gs.best_estimator_
    return clf, my_dataset

def get_importances():
  #changed final_features to feature_list
    feats_used = feature_list[1:]
    importances = clf.named_steps['clf'].feature_importances_
    for i, imp in enumerate(importances):
           print (i, feats_used[i], imp)

def features_used(features, clf):
    feat_bool = clf.named_steps['selector'].get_support()
    features = np.array(feature_list[1:])
    feats_used= features[feat_bool]
    return feats_used

def get_scores(feature_list,clf):
    feature_scores = []
    features = feature_list[1:]
    scores = clf.named_steps['selector'].scores_
    for i, x in enumerate(features):
      feature_scores.append((features[i], scores[i]))
    feature_scores = sorted(feature_scores, key=lambda feat_score: feat_score[1], reverse = True)
    feature_scores = pd.DataFrame(feature_scores, columns=["Features", "Scores"])
    feat_score_plot = sns.barplot(x=feature_scores["Features"], y=feature_scores["Scores"], color='b')
    plt.setp(feat_score_plot.get_xticklabels(), rotation=90)
    plt.ylabel("F-Score")
    plt.title("F-Score for each Feature")
    plt.tight_layout()
    plt.show()

    return feature_scores



def clean_data(data_dict):
    my_dataset = data_dict
    my_dataset.pop('BHATNAGAR SANJAY')
    my_dataset.pop('TOTAL')
    ef.email_poi_fraction(my_dataset)
    #check_NaN(my_dataset,total_features)
    #enron_only = remove_non_employees(my_dataset)
    return my_dataset



def check_poi_means():
    enron_data = featureFormat(my_dataset, total_features, sort_keys = True)
    enron_data = pd.DataFrame(enron_data)
    #print enron_data.head()
    enron_data.columns = total_features
    poi_non_poi_mean(enron_data,total_features)
#############
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature n

### The first feature must be "poi".
feature_list_main = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                     'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                     'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', "fraction_to_poi",
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'fraction_from_poi']

#feature list included all features plus fraction_to_poi will be reduced with select k best
feature_list_1d = ['poi','salary', 'bonus', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
                     'restricted_stock',  "fraction_to_poi",'shared_receipt_with_poi']

feature_list_2d = ['poi','total_stock_value', 'expenses', 'exercised_stock_options', 'other',
                     'restricted_stock',  "fraction_to_poi",'shared_receipt_with_poi']

feature_list_3d = ['poi', 'expenses', 'exercised_stock_options', 'other',  "fraction_to_poi",'shared_receipt_with_poi']

feature_list_4d = ['poi', 'expenses', 'exercised_stock_options', 'other','shared_receipt_with_poi']

feature_list_5d = ['poi', 'expenses', 'exercised_stock_options', 'other']


feature_list = ['poi', 'expenses', 'exercised_stock_options', 'other',  "restricted_stock",'fraction_to_poi']

feature_list_1a = ['poi', 'exercised_stock_options', 'other',  "restricted_stock",'fraction_to_poi']

feature_list_2a = ['poi', 'expenses', 'exercised_stock_options', 'other',  "restricted_stock"]


total_features = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                     'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                     'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
#all features minus , 'email_address'



with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#some interesting numbers that pop out, for shared_receipt, to_poi and from_poi it is more than double the amount for poi vs non_poi
#These could create a nice interation variable.

#all pois also have 0 loan advances, director fees, and restricted stock deffered

#bonus amount is double for poi and excercised stock options are triple.

#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
#features_used(my_features_list, clf)

tree = DecisionTreeClassifier(random_state=46)
forest = RandomForestClassifier(random_state=46)
supp_vec = SVC()
ada_boost =AdaBoostClassifier(random_state=46)


#clf = RandomForestClassifier(max_features= 5, n_estimators=100, min_samples_leaf=2, random_state=46)

cross_validator = StratifiedShuffleSplit(n_splits=100, test_size=0.1, train_size=None, random_state=46)


forest_param_grid = {'clf__max_features':(2,3),
                    'clf__min_samples_leaf':(1, 10)}

#tree_param_grid = {'clf__max_depth':(5,7, None),
#                   'clf__min_samples_split': (2, 4, 6, 10)}


#tree_param_grid = {'clf__max_depth':(5,7, None),
#                   'clf__min_samples_split': (2, 4, 6, 10)}

#tree_param_grid = {'clf__max_features':('auto', 2, 4, 5)}

#tree_param_grid = {'clf__max_depth':(5,7,8),
 #                  'clf__min_samples_split':(2, 4, 6, 10)}
tree_param_grid = {'clf__max_depth':(None, 20)}


svc_param_grid = {'clf__C': (1, 10),
                 'clf__gamma':(1, 10, 100),
                  'clf__kernel':('linear', 'rbf' )}

#adaboost_param_grid = {'clf__learning_rate':(.5, 1, 3),
#                    'clf__n_estimators':(10, 12, 50)}


#adaboost_param_grid = {'clf__learning_rate':(.5, 1, 2),
                   # 'clf__n_estimators':(7, 10, 12)}


adaboost_param_grid = {'clf__n_estimators':(12, 50)}



#clf, my_dataset= run_pipe(supp_vec, svc_param_grid)
#clf, my_dataset= run_pipe(tree, tree_param_grid)
#clf, my_dataset= run_pipe(forest, forest_param_grid)
#clf, my_dataset= run_pipe(ada_boost, adaboost_param_grid)



#table = kbest()
my_dataset = clean_data(data_dict)
clf, my_dataset = run_pipe(ada_boost, adaboost_param_grid)
#plot_kbest()
dump_classifier_and_data(clf, my_dataset, feature_list)


