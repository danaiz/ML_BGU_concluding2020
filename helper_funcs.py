import logging
import numpy as np
import pandas as pd
from random import sample
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.impute import KNNImputer as KNN
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import label_binarize

#instantiate both packages to use
encoder = OrdinalEncoder()
imputer = KNN()
# create a list of categorical columns to iterate over
def litlley(y):
    unique_y, counts = np.unique(y, return_counts= True)
    unique_ratios = counts/len(y)
    sorter = [(c,r) for c,r in sorted([*zip(unique_y, unique_ratios)], key=lambda x:x[1])]
    if sorter[0][1] < 0.1 and len(unique_y) > 2:
        y = np.where(y == sorter[0][0], sorter[1][0], y)
        y = litlley(y)
        return y
    else:
        return y
    

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

def all_incode(df):
#create a for loop to iterate through each column in the data
    cat = [i for i in df.columns if df.dtypes[i] in ['object','bool']]
    for col in cat:
        encode(df[col])
    df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)
    for col in cat:
        df[col] = np.round(df[col])
    return df

def norm_encode(y_true, y_pred):
    y_true_1 = set(y_true)
    y_pred_1 = set(y_pred)
    joint = y_true_1.copy()
    joint.update(y_pred_1)
    norm = OrdinalEncoder().fit_transform(np.array(list(joint)).reshape(-1,1))
    norm = norm.reshape(1,-1)[0]
    coding = {k:v for k,v in zip (np.array(list(joint)),norm)}
    y_true = [coding[i] for i in y_true]
    y_pred = [coding[i] for i in y_pred]
    return np.array(y_true).astype(int), np.array(y_pred).astype(int)

def weigted_tpr(y_true, y_pred):
    y_true, y_pred = norm_encode(y_true, y_pred)
    y_true_l = sorted(list(set(y_true)))
    y_pred_l = sorted(list(set(y_pred)))
    unique_y = np.unique(y_true, return_counts=True)[1]
    prop_y = unique_y/unique_y.sum()
    # if not len(y_pred_l) == len(y_true_l):
    position = set(y_pred_l).difference(set(y_true_l))
    for i in position:
        prop_y = np.insert(prop_y, i,0)
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    
    FN = FN.astype('float64')
    TP = TP.astype('float64')
    
    TPR = TP/(TP+FN)
    
    W_TPR = np.sum(np.where(np.isnan(TPR), 0, TPR)*prop_y)
    return W_TPR

def weigted_fpr(y_true, y_pred):
    y_true, y_pred = norm_encode(y_true, y_pred)
    y_true_l = sorted(list(set(y_true)))
    y_pred_l = sorted(list(set(y_pred)))
    unique_y = np.unique(y_true, return_counts=True)[1]
    prop_y = unique_y/unique_y.sum()
    # if not len(y_pred_l) == len(y_true_l):
    position = set(y_pred_l).difference(set(y_true_l))
    for i in position:
        prop_y = np.insert(prop_y, i,0)    
    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
    TP = np.diag(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    
    FN = FN.astype(float)
    FP = FP.astype(float)
    TN = TN.astype(float)
    FPR = FP/(FP+TN)
    W_FPR = np.sum(FPR*prop_y)
    return W_FPR

def buc(y_true, y_score):
    if len(np.unique(y_true)) == 1:
        if len(y_score.shape) == 1:
            num = 2
        else: 
            num = y_score.shape[1]
        diff = set(y_true).symmetric_difference(set(np.arange(num)))
        a = sample(diff,1)
        y_true = np.append(y_true, a)
        y_score = np.hstack((y_score, np.array(0.5)))
    try:
        score = metrics.roc_auc_score(y_true, y_score)
    except:
        try:
            score = metrics.roc_auc_score(y_true, y_score[:,1],labels = np.arange(y_score.shape[1]))
        except:
            score = metrics.roc_auc_score(y_true, y_score ,labels = np.arange(y_score.shape[1]), multi_class = 'ovr')
    return score

def ap(y_true, y_score):
    if len(np.unique(y_true)) == 1:
        if len(y_score.shape) == 1:
            num = 2
        else: 
            num = y_score.shape[1]
        diff = set(y_true).symmetric_difference(set(np.arange(num)))
        a = sample(diff,1)
        y_true = np.append(y_true, a)
        y_score = np.hstack((y_score, np.array(0.5)))
    try:
        ap_score = metrics.average_precision_score(y_true, y_score, average='weighted')
    except: 
        Y_true = label_binarize(y_true, classes=np.arange(y_score.shape[1]))
        ap_score = metrics.average_precision_score(Y_true, y_score, average='weighted')
      
    return ap_score



weighted_P = make_scorer(metrics.precision_score, average = 'weighted')
weighted_R =  make_scorer(metrics.recall_score, average = 'weighted')
accuracy = make_scorer(metrics.accuracy_score)
auc = make_scorer(buc, needs_proba=True)
wtpr = make_scorer(weigted_tpr)
wfpr = make_scorer(weigted_fpr)
AP = make_scorer(ap,needs_proba=True)

