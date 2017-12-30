function entropy = calculate_entropy(Y)
    % Function to calculate entropy of Y label
    p_times_logp = @(x) min(0, x.*log2(x));
    count = zeros(1,5);
    total = size(Y,1);
    entropy = 0;
    for i = 1:5
        count(1,i) = size(Y(Y==i),1);
        frac = count(1,i)/total;
        entropy = entropy-p_times_logp(frac);
    end
end