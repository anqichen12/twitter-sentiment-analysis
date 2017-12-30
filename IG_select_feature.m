function [idx,select_data_ig,vocabulary_ig] = IG_select_feature(X,Y,V,num)

    % Function to implement the feature selection method to reduce dimension.
    % Usage: idx: num x 1 matrix selected index
    %        select_data_ig: N x num data matrix after dimension reduction,
    %        where N is number of data points in X_train_bag
    %        vocabulary_ig: 1 x num matrix selected vocabulary 
    %        num: dimension number selected

    ig = zeros(size(X,2),1);
    % get H_c: class entropy
    H_c = calculate_entropy(Y);
    % loop through every word
    for j = 1:size(X,2)
        Y_1 = Y(X(:,j)>0);
        Y_0 = Y(X(:,j)==0);
        p_Y1 = size(Y_1,1)/size(Y,1);
        p_Y0 = size(Y_0,1)/size(Y,1);
        H_Y1 = calculate_entropy(Y_1);
        H_Y0 = calculate_entropy(Y_0);
        ig(j,1) = H_c - p_Y1*H_Y1 - p_Y0*H_Y0;
    end
    % sort and select top index which have maximum information gain
    [sortedX, sortedInds] = sort(ig,'descend');
    idx = sortedInds(1:num);
    select_data_ig = X(:,idx(:,1));
    vocabulary_ig = V(:,idx(:,1));
end