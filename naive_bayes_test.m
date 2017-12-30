function [class] = naive_bayes_test(Xtrain,Ytrain,Xtest,cost_matrix)
%Predict label using naive bayes
%   Output - class: t x 1 predicted labels for test data
%   Input  - Xtrain: m x n training data set
%          - Ytrain: m x 1 label for the training data
%          - Xtest: t x n test data
%          - cost_matrix: 5 x 5 cost matrix for cost sensitive classification
    nb = fitcnb(Xtrain,Ytrain,'Distribution','mn','Cost',cost_matrix,'HyperparameterOptimizationOptions','all');
    [class,Posterior,Cost] = predict(nb,Xtest);
end