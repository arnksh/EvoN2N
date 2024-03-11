
load('DataForClassification_TimeDomain.mat')
seg = 400;
data = AccTimeDomain';
DataS = []; DataT = [];
for i = 1:9
    for j=104*(i-1)+1:104*(i-1)+83
        DataS = [DataS; reshape(data(i,:), 3600/seg, seg)];
    end
    for j=104*(i-1)+84:104*(i-1)+104
        DataT = [DataT; reshape(data(i,:), 3600/seg, seg)];
    end
end

labelS = zeros((3600/seg)*83,1);
labelT = zeros((3600/seg)*21,1);
for i=1:8
    labelS = [labelS; i*ones((3600/seg)*83,1)];
    labelT = [labelT; i*ones((3600/seg)*21,1)];
end
Y = train_test_val(DataT, labelT);
save('GFD', 'Y')
Y = train_test(DataS, labelS);
save('GFD_s', 'Y')
