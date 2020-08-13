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

### using SMOTEBoost.py

