function [Y_hat] = predict_labels(X_test_bag, test_raw, choosemodel)


% Inputs:   X_test_bag     n x 10000 bag of words features
%           test_raw      n x 1 cells containing all the raw tweets in text
%           approach      1   logistic regression
%                         2   naive bayes
%                         3   knn
    load('train.mat')
    load('cov_matrix.mat');
    load('vocabulary.mat');
    if choosemodel == 1
        [idx,xtrain,vocabulary_ig] = IG_select_feature(X_train_bag,Y_train,vocabulary,9000);
        xtest = X_test_bag(:,idx);
        [Y_hat] = log_regression_test(xtest,xtrain,Y_train,cost_matrix);
    elseif choosemodel == 2
        [idx,xtrain,vocabulary_ig] = IG_select_feature(X_train_bag,Y_train,vocabulary,9000);
        xtest = X_test_bag(:,idx);
        [Y_hat] = naive_bayes_test(xtrain,Y_train,xtest,cost_matrix);
    elseif choosemodel == 3
        [idx,xtrain,vocabulary_ig] = IG_select_feature(X_train_bag,Y_train,vocabulary,2000);
        xtest = X_test_bag(:,idx);
        [Y_hat] = knn_test(xtrain,Y_train,xtest,cost_matrix);
    else
        fprintf("wrong model number");
    end
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

end
