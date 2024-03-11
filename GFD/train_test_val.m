function [Y] = train_test_val(Data, labels)


TF = isnan(Data);
Data(TF) = 0;
Data = mn_var_norm(Data);

if min(labels~=0)
    labels = labels-1;
end

k=5;
c=cvpartition(length(labels),'kfold',k);
x=Data(training(c,1),:);
y=labels(training(c,1),:);
x_test=Data(test(c,1),:); %keep 20% of total samples for testing
y_test=labels(test(c,1),:);


c = cvpartition(length(y), 'kfold', 5);
x_train=x(training(c,1),:);
y_train=y(training(c,1),:);
x_val=x(test(c,1),:);  % Keeps 20% of training data(80% of total samples)
y_val=y(test(c,1),:);

Y.training_inputs = x_train; Y.training_results = y_train;
Y.test_inputs = x_test; Y.test_results = y_test;
Y.val_inputs = x_val; Y.val_results = y_val;


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
