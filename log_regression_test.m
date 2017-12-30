function [Y_hat] = log_regression_test(xtest,xtrain,Y_train,cost_matrix)
%Predict the label using logsitic regression
%   Output - Y_hat: t x 1 predicted labels for test data
%   Input  - xtrain: m x n training data set
%          - ytrain: m x 1 label for the training data
%          - xtest: t x n test data
%          - cost_matrix: the cost matrix for cost sensitive classification
    model = train(Y_train, xtrain, '-s 0 -c 2 -w1 0.2426 -w2 0.3567 -w3 0.0868 -w4 0.1068 -w5 0.2071', 0);
    nolabel = zeros(size(xtest,1),1);
    [predicted_label, accuracy, prob_estimates] = predict(nolabel, xtest, model, '-b 1',0);
    id = model.Label;
    bb = prob_estimates(:,id);
    [max_a,Y_hat]=min(bb*cost_matrix,[],2);
end

