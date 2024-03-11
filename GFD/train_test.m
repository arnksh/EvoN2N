function [Y] = train_test(Data, labels)


TF = isnan(Data);
Data(TF) = 0;
Data = mn_var_norm(Data);

if min(labels~=0)
    labels = labels-1;
end

k=5;
c=cvpartition(length(labels),'kfold',k);
x_train=Data(training(c,1),:);
y_train=labels(training(c,1),:);
x_test=Data(test(c,1),:); %keep 20% of total samples for testing
y_test=labels(test(c,1),:);


Y.training_inputs = x_train; 
Y.training_results = y_train;
Y.test_inputs = x_test; 
Y.test_results = y_test;
end


function [norm_data, mn, std_dev] = mn_var_norm(data, mn, std_dev)

if nargin == 1
    mn = mean(data,1);
    std_dev = std(data);
end

for j = 1:size(data,2)
    if std_dev(j) == 0
        norm_data(:,j) = zeros(size(data,1), 1);
        
    else
        norm_data(:,j) = (data(:,j) - repmat(mn(j), size(data,1), 1)) ./ repmat(std_dev(j), size(data,1), 1);
    end
end
end
