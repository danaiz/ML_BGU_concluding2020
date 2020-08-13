import os
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
from math import ceil
from timeit  import timeit
from helper_funcs import *
from SMOTEBoost import SMOTEBoost
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,cross_validate, KFold

def merge_dict(d1, d2):
    ds = [d1, d2]
    d = {}
    for k in d1.keys():    
        d[k] = np.concatenate(list(d[k] for d in ds))
    return d
def run_analysis(dfdir = None,
                 alg_name = None,
                 params_dict = None,
                 output_filname = 'summary_df', 
                 clf = None):
    
    first_flag = True
    final_dict = {}
    exceptions = []
    for dfname in os.listdir(dfdir):
        if dfname.endswith('csv'):
           logging.info("##### New DF ##### \n\t\t\t\t\t\t\t\t\t\t\t\t\t\t Df name: {}".format(dfname))
           li = LabelEncoder()
           df = pd.read_csv(os.path.join(dfdir,dfname))
           X = df.iloc[:,:-1]
           cat = [i in ['object','bool'] for i in X.dtypes]
           X = all_incode(X)
           X = pd.get_dummies(X, columns=X.columns[cat].tolist())
           y = litlley(df.iloc[:,-1].values)
           y =  li.fit_transform(y)
           try:
               meas_dict = fit_output(clf, params_dict, X, y, dfname, alg_name)
           except Exception as e:
               logging.info(e)
               exceptions.append(dfname)
               continue
           if first_flag:
               final_dict = meas_dict
               first_flag = False
           else:
               final_dict = merge_dict(final_dict, meas_dict) 
    pd.DataFrame(final_dict).to_csv('{}.csv'.format(output_filname), index = False)
    pd.Series(exceptions).to_csv('exceptions_{}'.format(len(exceptions)))
def fit_output(clf, params_dict, X, y, filename, alg_name):
    global rs
    global thousand
    measures = {}
    scoring = {
            'accuracy': accuracy,
            'TPR': wtpr,
            'FPR' :wfpr,
            'AUC' : auc,
            'Precision': weighted_P,
            'Recall':weighted_R,
            'PR curve':AP
            }
    if clf is RandomForestClassifier:
        est = clf()
    else:
        est = OneVsRestClassifier(clf(), n_jobs = -1)
    logging.info('\tRunning RandomizedSearch...')
    rs = RandomizedSearchCV(est, params_dict, cv = 3, refit = True, n_iter = 50,
                            n_jobs = -1).fit(X,y)  
    logging.info('\t\tfitted best estimator') 
    new_params = rs.best_estimator_.get_params()
    params_string = ', '.join('{0} = {1}'.format(k.replace('estimator__',''),
                                            v) for k,v in new_params.items())
    
    if clf is RandomForestClassifier:
        cvest = clf().set_params(**new_params)
    else:
        cvest = OneVsRestClassifier(clf(), n_jobs=-1).set_params(
                **new_params)
    
    try:
        measures['train_time'] = np.array([rs.refit_time_]*10)
        measures['Parameters'] = np.array([params_string]*10)
        measures['Dataset name'] = np.array([filename]*10)
        measures['Algorithm name'] = np.array([alg_name]*10)
        measures['Cross validation'] = np.arange(1,11)
        outer_cv = KFold(n_splits=10, shuffle=True)
        logging.info('\tRunning Cross Validation...')
        cvfit = cross_validate(cvest, X, y, cv=outer_cv, n_jobs=-1, scoring = scoring)
        measures.update(cvfit)
        logging.info('\t\t Done with cross Validation...')
        if len(X) >= 1000: 
            thousand = X.sample(1000)
        else:
            thousand = X.sample(1000, replace = True)
        measures['Inference Time'] = np.array([timeit(
            'rs.predict(thousand)' ,setup = 'from __main__ import rs, thousand',
               number = 3)/3]*10)
    except Exception as e:
        logging.info(e)
        measures['train_time'] = np.array([rs.refit_time_]*3)
        measures['Parameters'] = np.array([params_string]*3)
        measures['Dataset name'] = np.array([filename]*3)
        measures['Algorithm name'] = np.array([alg_name]*3)
        measures['Cross validation'] = np.arange(1,4)
        outer_cv = KFold(n_splits=3, shuffle=True)
        logging.info('\tRunning Cross Validation...')
        cvfit = cross_validate(cvest, X, y, cv=outer_cv, n_jobs=-1, scoring = scoring)
        measures.update(cvfit)
        logging.info('\t\t Done with cross Validation...')
        if len(X) >= 1000: 
            thousand = X.sample(1000)
        else:
            thousand = X.sample(1000, replace = True)
        measures['Inference Time'] = np.array([timeit(
            'rs.predict(thousand)' ,setup = 'from __main__ import rs, thousand',
               number = 3)/3]*3)
    logging.info('!!!!!Finished with {}.\n!!!!!'.format(filename))
    return measures



if __name__ == '__main__':
     logging.basicConfig(filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%d-%m %H:%M:%S',
                            filename = 'Randomlog.log',
                            level=logging.INFO)
     
     logging.info("Helo handsome;)")

     parser = argparse.ArgumentParser()
     
     parser.add_argument('-d', '--dir')
     parser.add_argument('-a', '--alg_name')
     parser.add_argument('-p', '--params_dict_path')
     parser.add_argument('-o', '--output_filname', default = 'summary_df')
     parser.add_argument('-c', '--clf')
     
     args = parser.parse_args()
     
     exec('clf='+args.clf)
     params = pickle.load(open(args.params_dict_path, 'rb'))
     
     test_dict = run_analysis(dfdir = os.path.join(os.getcwd(), args.dir),
                 alg_name = args.alg_name,
                 params_dict = params,
                 output_filname = args.output_filname, 
                 clf =  clf)