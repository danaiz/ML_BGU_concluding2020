import shap
import itertools 
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder,StandardScaler

rf = pd.read_csv('results/RandomForest_summary.csv')
sb = pd.read_csv('results/SMOTE_summary.csv')

metaf = pd.read_csv('ClassificationAllMetaFeatures.csv')
metaf = metaf.replace(['abalone', 'statlog-heart','wine_quality_red','pittsburg-bridges-T-OR-D' ],
                                      ['abalon', 'statlog-heart_', 'wine-quality-red', 'pittsburg-bridges-T-OR-D_R']).set_index('dataset')

def process(df):
    ser = df.groupby('Dataset name').agg({'test_AUC':'mean'})
    ser.index =  ser.index.str.replace('.csv','')
    return ser

def rank(rf, sb):
    aucs = pd.merge(rf, sb, left_index=True, right_index=True).\
    rename(columns={'test_AUC_x':'RandomForest','test_AUC_y':'SMOTEBoost'})
    
    for row in aucs.itertuples():
        if row.RandomForest>row.SMOTEBoost:
            aucs.loc[row.Index, 'RandomForest'] = 1
            aucs.loc[row.Index, 'SMOTEBoost'] = 0
        elif row.RandomForest<row.SMOTEBoost:
            aucs.loc[row.Index, 'RandomForest'] = 0
            aucs.loc[row.Index, 'SMOTEBoost'] = 1
        else:
            aucs.loc[row.Index, 'RandomForest'] =  aucs.loc[row.Index, 'SMOTEBoost'] = 1
    rf = pd.DataFrame(aucs.RandomForest).rename(columns = {'RandomForest':'won'})
    sb = pd.DataFrame(aucs.SMOTEBoost).rename(columns = {'SMOTEBoost':'won'})
    return rf,sb
def mercon(rf, sb, meta):
    meta1 = pd.merge(meta, rf, left_index=True, right_index=True)
    meta2 = pd.merge(meta, sb, left_index=True, right_index=True)
    metcon = pd.concat([meta1, meta2], axis = 0)
    return metcon

def XGB_gridsearch(xg_train, gridsearch_params, params, change = None):
    max_auc = 0
    best_params = None
    for param1, param2 in gridsearch_params:
        # Update parameters
        params[change[0]] = param1
        params[change[1]] = param2
        # Run CV
        cv_results = xgb.cv(
            params,
            xg_train,
            num_boost_round=num_boost_round,
            nfold=3,
            metrics={'auc'},
            early_stopping_rounds=early_stopping
        )
        # Update best MAE
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds = cv_results['test-auc-mean'].idxmax()
        print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = (param1,param2)
    print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))
    return best_params
       
rf = process(rf)
sb = process(sb)
rf,sb = rank(rf, sb)
rf['algorithm'] = 'RandomForest'
sb['algorithm'] = 'SMOTEBoost'
meta_n = mercon(rf, sb, metaf)

li = LabelEncoder()
meta_n.algorithm = li.fit_transform(meta_n.algorithm)

X = meta_n.drop('won', axis = 1)
y = meta_n.won

## PARAMS for XGBoost : 
num_boost_round = 999
n_folds = 3
early_stopping = 15
## GridSearch PARAMS for XGBoost
params = {'eta': 0.02,
          'max_depth': 1,
          'subsample': 1,
          'colsample_bytree': 1,
          'objective': 'binary:logistic',
          'eval_metric':'auc',
          'nthread':8,
          'min_child_weight': 1}

xg_train = xgb.DMatrix(X, label = y);
max_depth = np.arange(2,12)
min_child_weight = np.append(np.linspace(0,1,10), np.linspace(2,8,7))
gridsearch_params =  list(itertools.product(max_depth, min_child_weight))

##############################################################
### GridSearch for 2 hyperparamaters :
### Max_Depth and Min_Child_Weight
###
##############################################################
best_params = XGB_gridsearch(xg_train, gridsearch_params,
               params, change = ['max_depth','min_child_weight'])
params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]

subsample = np.linspace(0,1,10)    
colsample = np.linspace(0,1,10)   
gridsearch_params =  list(itertools.product(subsample, colsample))
##############################################################
### GridSearch for 2 hyperparamaters :
### SubSample and Col_Sample_ByTree
###
##############################################################
best_params = XGB_gridsearch(xg_train, gridsearch_params,
               params, change = ['subsample','colsample_bytree'])
params['subsample'] = best_params[0]
params['colsample_bytree'] = best_params[1]

##############################################################
### GridSearch for 2 hyperparamaters :
### Gamma, Lambda 
###
##############################################################
gamma = np.append(np.linspace(0,1,10), np.linspace(2,8,7))
lambd = np.arange(1,5)
gridsearch_params =  list(itertools.product(gamma, lambd))
best_params = XGB_gridsearch(xg_train, gridsearch_params,
               params, change = ['gamma','lambda'])
params['gamma'] = best_params[0]
params['lambda'] = best_params[1]
##############################################################
### GridSearch for 1 hyperparamaters :
### ETA : Learning Rate. 
###
##############################################################

max_auc = 0
best_params = None
for eta in [.3, .2, .1, .05, 0.02,.01, .005]:
    # update our parameters
    params['eta'] = eta
    # CV
    cv_results = xgb.cv(
            params,
            xg_train,
            num_boost_round=num_boost_round,
            nfold=n_folds,
            metrics=['auc'],
            early_stopping_rounds=early_stopping)
    # Update best score
    mean_auc = cv_results['test-auc-mean'].max()
    boost_rounds = cv_results['test-auc-mean'].idxmax()
    print("\tAUC {} for {} rounds\n".format(mean_auc, boost_rounds))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = eta
print("Best params: {}, AUC: {}".format(best_params, max_auc))
params['eta'] = best_params
params['eval_metric'] = 'error'

loo = LeaveOneOut()
right = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]
    xg_train = xgb.DMatrix(X_train, label = y_train)
    xg_test = xgb.DMatrix(X_test, label = y_test)
    best_model = xgb.train(
    params,
    xg_train,
    evals = [(xg_test, 'test')],
    num_boost_round=100,
    verbose_eval=False)
    
    if np.where(best_model.predict(xg_test) > 0.5, 1,0) == y_test:
        right.append(1)
    else:
        right.append(0)
print('Acc is {} by Leave one out evaluation'.format(sum(right)/len(right)))

best_model_final = xgb.train(
    params,
    xgb.DMatrix(X, label = y),
    num_boost_round=100,
    verbose_eval=False)

fig,ax = plt.subplots(figsize = (20,10), constrained_layout=True)    
xgb.plot_importance(best_model_final,ax=ax,importance_type = 'cover',
                    title = "Feature importance - Cover")
fig.savefig('cover.jpeg')

fig1,ax1 = plt.subplots(figsize = (20,10), constrained_layout=True)    
xgb.plot_importance(best_model_final,ax=ax1,importance_type = 'gain',
                    title = "Feature importance - Gain")
fig1.savefig('gain.jpeg')

fig2,ax2 = plt.subplots(figsize = (20,10), constrained_layout=True) 
xgb.plot_importance(best_model_final,ax=ax2,importance_type = 'weight',
                    title = "Feature importance - Weight")
fig2.savefig('weight.jpeg')

explainer = shap.TreeExplainer(best_model_final)
shap_values = explainer.shap_values(X)
fig4,ax4 = plt.subplots(constrained_layout=True) 
shap.summary_plot(shap_values, X, plot_type = 'dot', show = False, plot_size = (20,10))
fig4.savefig('shap.jpeg')

#### Saving the model :
# best_model.save_model("xgb_model.model")

