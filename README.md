# Cocluding assignment
### Dan Aizenberg and Yuval Tamir - Machine learning, BGU, 2020

### using SMOTEBoost.py
SMOTEBoost.py is based on and inharits scikit-learn properties. That means the it has three main functions:

 1. initializtion: all you need to do is call the function and pass it's parameters. for example: SMOTEBoost(k_neighbors=5, n_estimators=4). The relevant hyperparameters are:
	 - k_neighbors - number of neighbors for SMOTE algorithm.
	 - n_estimators - number of weak learners to train.
	 - base_estimator - base_estimator
	 - learning_rate - learnig rate for every boosting step
	 - algorithm - AdaBoost Algorithm to use
2. Training: training is fone by the method fit(X,y). X is the unlabeld training data, and y is the lables.
3. Predicting: You can either predict label by using predict() method and passing the unlabeld cases to predict, or predicting the probabilities for each class by using predict_proba() methond and passing the unlabeld cases to predict.
4. All other relevant scikit-learn's methods are available too.

### using ML_automation.py
ML_automation.py is made for command line use. it returns a CSV file with the folloeing parameters for the classifier and datasets you provide:
- Dataset name
- Algorithm name
- Cross validation round
- Hyper-parameters used
- Accuracy
- TPR
- FPR
- Precision
- AUC
- PR-Curve AUC
- Training time
- Inference Time

It does so by running a randommized search on a predetermind set of hyper-parameters for each dataset with the chosen algorithm. It then saves the results and outputsthem all together in one table.

**How to use?**
In your command line, call python and the file eith the following flags:

    -d  - the directory in which all  the datasets needed for the analysis are located and only them.
    -a - the name of the algorithm to use
    -p a pickled dictionary with the relevant paramerters for Randomized search. the format can be found here

**
