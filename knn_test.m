function [Y_hat_knn] = knn_test(Xtrain,Ytrain,Xtest,cost_matrix)
%Predict the label using k nearest neighbor
%   Output - Y_hat_knn: t x 1 predicted labels for test data
%   Input  - Xtrain: m x n training data set
%          - Ytrain: m x 1 label for the training data
%          - Xtest: t x n test data
%          - cost_matrix: 5 x 5 cost matrix for cost sensitive classification
    Mdl = fitcknn(Xtrain,Ytrain,'NumNeighbors',5,'Cost',cost_matrix,'distance','jaccard');
    [Y_hat_knn,score,cost] = predict(Mdl,Xtest);
end