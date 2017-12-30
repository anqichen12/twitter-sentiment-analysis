Used soft-voting model combining logistic regression and naive bayes to predict twitter sentiments

====usage====

Information gain is used to reduce dimension. Three models are trained, which are logistic regression, naive bayes and k nearest neighbor.

First, run calculate_entropy.m, IG_select_feature.m, knn_test.m, log_regression_test.m, naive_bayes_test.m. Then, run main function predict_labels.m.

[Y_hat] = predict_labels(X_test_bag, test_raw, choosemodel)

Input   - X_test_bag m x 10000 bag of words features
	- test_raw m x 1 cells containing all the raw tweets in text
	- choosemodel There are three model can be trained to get Y_hat
	  if choosemodel = 1, trained logistic regression classifier
	  if choosemodel = 2, trained naive bayes classifier 
	  if choosemodel = 3, trained knn classifier
	
Output -  Y_hat m x 1 predicted labels given by corresponding classifier

Feature selection - select top 9000 features sorted by information gain for logistic 	  
                    regression and naive bayes classifier
		  - select top 2000 features sorted by information gain for knn classifier
