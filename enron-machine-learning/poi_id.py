#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester


from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


### Task 1: Select what features you'll use.
# These are the features in the enron dataset.
# I will create new features later and
# select best features according to algorithm
# using SelectKBest().



target_label = 'poi'
email_features_list = [ 
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]
features_list = [target_label] + financial_features_list + email_features_list








### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r"))


### Task 2: Remove outliers



### Remove outliers
data_dict.pop('TOTAL') # not a person
data_dict.pop('THE TRAVEL AGENCY IN THE PARK') # not a person
data_dict.pop('LOCKHART EUGENE E') # no data


### Task 3: Create new feature(s)




def fraction_to_poi_email(data_dict):
    features = [ 'from_messages', 'from_this_person_to_poi']

    for key in data_dict:
        name = data_dict[key]

        is_null = False
        for feature in features:
            if name[feature] == 'NaN':
                is_null = True

        if not is_null:
            name['fraction_to_poi_email'] = float(name['from_this_person_to_poi'] ) /\
                                                 ( name['from_messages'])
        else:
            name['fraction_to_poi_email'] = 'NaN'


def total_net_worth(data_dict):
    features = ['salary', 'bonus', 'total_stock_value', 'exercised_stock_options']

    for key in data_dict:
        name = data_dict[key]

        is_null = False
        for feature in features:
            if name[feature] == 'NaN':
                is_null = True

        if not is_null:
            name['total_net_worth'] = name['salary'] + name['total_stock_value'] +\
                                    name['bonus'] + name['exercised_stock_options']
        else:
            name['total_net_worth'] = 'NaN'



fraction_to_poi_email(data_dict)
total_net_worth(data_dict)

features_list += ['fraction_to_poi_email', 'total_net_worth']




### Store to my_dataset for easy export below.
my_dataset = data_dict



def get_k_best(data_dict, features_list, k):

    data = featureFormat(data_dict, features_list)
    labels_train, features_train = targetFeatureSplit(data)

    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(features_train, labels_train)

    unsorted_list = zip(features_list[1:], k_best.scores_)
    sorted_list = sorted(unsorted_list, key=lambda x: x[1], reverse=True)
    k_best_features = dict(sorted_list[:k])

    return ['poi'] + k_best_features.keys()




# Top 10 best features using SelectKBest().
best_features_list = ['poi',
                      'total_stock_value',
                      'long_term_incentive',
                      'bonus',
                      'fraction_to_poi_email',
                      'restricted_stock',
                      'total_net_worth',
                      'expenses',
                      'shared_receipt_with_poi',
                      'deferred_income',
                      
                       ]


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


### Task 4: Tune and try a variety of classifiers

def tune_logistic_regression():

    skb = SelectKBest()
    pca = PCA()
    lr_clf = LogisticRegression()

    pipe_lr = Pipeline(steps=[("SKB", skb), ("PCA", pca), ("LogisticRegression", lr_clf)])

    lr_k = {"SKB__k": range(9, 10)}
    lr_params = {'LogisticRegression__C': [1e-08, 1e-07, 1e-06],
                 'LogisticRegression__tol': [1e-2, 1e-3, 1e-4],
                 'LogisticRegression__penalty': ['l1', 'l2'],
                 'LogisticRegression__random_state': [42, 46, 60]}
    lr_pca = {"PCA__n_components": range(3, 8), "PCA__whiten": [True, False]}

    lr_k.update(lr_params)
    lr_k.update(lr_pca)

    get_best_parameters_reports(pipe_lr, lr_k, features, labels)

    

def tune_svc():

    skb = SelectKBest()
    pca = PCA()
    svc_clf = SVC()

    pipe_svc = Pipeline(steps=[("SKB", skb), ("PCA", pca), ("SVC", svc_clf)])

    svc_k = {"SKB__k": range(8, 10)}
    svc_params = {'SVC__C': [1000], 'SVC__gamma': [0.001], 'SVC__kernel': ['rbf']}
    svc_pca = {"PCA__n_components": range(3, 8), "PCA__whiten": [True, False]}

    svc_k.update(svc_params)
    svc_k.update(svc_pca)

    get_best_parameters_reports(pipe_svc, svc_k, features, labels)


def tune_decision_tree():

    skb = SelectKBest()
    pca = PCA()
    dt_clf = DecisionTreeClassifier()

    pipe = Pipeline(steps=[("SKB", skb), ("PCA", pca), ("DecisionTreeClassifier", dt_clf)])

    dt_k = {"SKB__k": range(8, 10)}
    dt_params = {"DecisionTreeClassifier__min_samples_leaf": [2, 6, 10, 12],
                 "DecisionTreeClassifier__min_samples_split": [2, 6, 10, 12],
                 "DecisionTreeClassifier__criterion": ["entropy", "gini"],
                 "DecisionTreeClassifier__max_depth": [None, 5],
                 "DecisionTreeClassifier__random_state": [42, 46, 60]}
    dt_pca = {"PCA__n_components": range(4, 7), "PCA__whiten": [True, False]}

    dt_k.update(dt_params)
    dt_k.update(dt_pca)

    get_best_parameters_reports(pipe, dt_k, features, labels)









if __name__ == '__main__':

    '''         GAUSSIAN NAIVE BAYES            '''

    clf = GaussianNB()
    print "Gaussian Naive Bayes : \n", tester.test_classifier(clf, my_dataset, best_features_list)


    '''         LOGISTIC REGRESSION             '''

    #tune_logistic_regression()

    best_features_list_lr = get_k_best(my_dataset, features_list, 9)

    clf_lr = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=4, whiten=False)),
        ('classifier', LogisticRegression(tol=0.01, C=1e-08, penalty='l2', random_state=42))])

    print "Logistic Regression : \n", tester.test_classifier(clf_lr, my_dataset, best_features_list_lr)


    '''         SUPPORT VECTOR CLASSIFIER           '''

    #tune_svc()

    best_features_list_svc = get_k_best(my_dataset, features_list, 8)

    clf_svc = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=6, whiten=True)),
        ('classifier', SVC(C=1000, gamma=.001, kernel='rbf'))])

    print "Support Vector Classifier : \n", tester.test_classifier(clf_svc, my_dataset, best_features_list_svc)


    '''         DECISION TREE CLASSIFIER            '''

    #tune_decision_tree()

    best_features_list_dt = get_k_best(my_dataset, features_list, 8)

    clf_dt = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=5, whiten=True)),
        ('classifier', DecisionTreeClassifier(criterion='entropy',
                                              min_samples_leaf=2,
                                              min_samples_split=2,
                                              random_state=46,
                                              max_depth=None))
    ])

    print "Decision Tree Classifier : \n",tester.test_classifier(clf_dt, my_dataset, best_features_list_dt)









    '''         dump final algorithm classifier, dataset and features in the data directory         '''

dump_classifier_and_data(clf_lr, my_dataset, best_features_list_lr)




# evaluation of algorithm:

def get_best_parameters_reports(clf, parameters, features, labels):


    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=60)


    cv_strata = StratifiedShuffleSplit(labels_train, 100, test_size=0.2, random_state=60)

    grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=cv_strata, scoring='f1')
    grid_search.fit(features_train, labels_train)

    '''
    prediction = grid_search.predict(features_test)
    print 'Precision:', precision_score(labels_test, prediction)
    print 'Recall:', recall_score(labels_test, prediction)
    print 'F1 Score:', f1_score(labels_test, prediction)


    '''

    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
