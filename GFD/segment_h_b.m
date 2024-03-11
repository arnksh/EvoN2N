clear all
nameOfFault = {'Healthy', 'BrokenTooth'};
fi = {'h', 'b'};

Data=[];
labels = [];
lod = 0;
for fault=1:length(nameOfFault)
    % Enter the path
    folder=nameOfFault{fault};
    data = csvread(['./' folder '/' fi{fault} '30hz' int2str(lod) '.csv'], 1, 0);
    Data_segmented = reshape(data(1:80000), [800 100]);
    Data=[Data; Data_segmented];
    labels = [labels;(fault-1)*ones(size(Data_segmented,1),1)];
end

Y = train_test(Data, labels);
save(['h_b_30hz_' int2str(lod)], 'Y')
%%

for lod = 10:10:90
    Data=[];
labels = [];
for fault=1:length(nameOfFault)
    % Enter the path
    folder=nameOfFault{fault};
    data = csvread(['./' folder '/' fi{fault} '30hz' int2str(lod) '.csv'], 1, 0);
    Data_segmented = reshape(data(1:30000), [300 100]);
    Data=[Data; Data_segmented];
    labels = [labels;(fault-1)*ones(size(Data_segmented,1),1)];
end

Y = train_test_val(Data, labels);

save(['h_b_30hz_' int2str(lod)], 'Y')
end
