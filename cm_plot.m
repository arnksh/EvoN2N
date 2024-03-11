clear all; clc
addpath ./ReqFnNSGAII/;  
dataFolder = '../PBU_400';
srcData  = 'sourceData';
tarData = {'tarData_1','tarData_2','tarData_3','tarData_4'};
%% plot signals source data
load([dataFolder '/' srcData], 'Data');
figure
plot(Data(:,1), '-r');
hold on
plot(Data(:,3), '-m');
plot(Data(:,4),'-b');
plot(Data(:,2),'-b');
% ylim([-10, 10])
set(gca,'XTick',[], 'YTick', [])
legend('Sig_1', 'Sig_2', 'Sig_3')
% legend boxoff

%% COnfusion matrix
clear all;
clc
dataFolder = 'CWRU_400';
tarData = {'FE_tar_7_1', 'FE_tar_7_2', 'FE_tar_7_3', 'FE_tar_14_1', 'FE_tar_14_2',...
    'FE_tar_14_3', 'FE_tar_21_1', 'FE_tar_21_2', 'FE_tar_21_3'};

     load(['../' dataFolder '/' tarData{1}], 'Y');
     T3 = readmatrix("results_EvoN2N.xlsx","Sheet",1,"Range",'E5:M5'); % D28:K28 for PBU, D43:K43 for GFD

     methods = {'SVM', 'DNN', 'DANN', 'DTL','DAFD', 'N2N\_WDA', 'N2N\_DA', 'EvoDCNN', 'EvoN2N'};


    % xt = Y.test_inputs';
    yt = Y.test_results';

    figure
    for i= 1:length(methods)
        subplot(3,3,i)
        yprd = tr2pred(yt, T3(i));
        confusionchart(yt, yprd, "FontSize", 12, 'Title', methods{i}, 'DiagonalColor','#808080');
    end
    % subplot(3,3,9)
    % txt = sprintf('0 = Normal Running  \n1 = Inner Fault (IR) \n2 = Ball Fault (B) \n3 = Outer Race Fault (OR)');
    % text(0.5, 0.5, txt, 'HorizontalAlignment', 'center', 'FontSize', 12);
    % axis off

%% COnfusion matrix
clear all;
clc
dataFolder = 'PBU_400';
tarData = {'tarData_1','tarData_2','tarData_3','tarData_4','tarData_5'};

     load(['../' dataFolder '/tar' int2str(3) '/' tarData{1}], 'Y');

     T3 = readmatrix("results_EvoN2N.xlsx","Sheet",1,"Range",'E33:M33'); % D28:K28 for PBU, D43:K43 for GFD

     methods = {'SVM', 'DNN', 'DANN', 'DTL','DAFD', 'N2N\_WDA', 'N2N\_DA', 'EvoDCNN', 'EvoN2N'};


    yt = Y.test_results';

    figure
    for i= 1:length(methods)
        subplot(3,3,i)
        yprd = tr2pred(yt, T3(i));
        confusionchart(yt, yprd, "FontSize", 12, 'Title', methods{i}, 'DiagonalColor','#808080');
    end
    % subplot(3,3,9)
    % txt = sprintf('0 = Normal Running  \n1 = Inner Fault (IR) \n2 = Ball Fault (B) \n3 = Outer Race Fault (OR)');
    % txt = sprintf('0 = Healthy/Normal(N)  \n1 = Outer Race (OR) \n2 = Inner Race (IR)');
    % txt = sprintf('0 = Healthy/Normal(N)  \n1 = Broken Tooth (BT)');
    % text(0.5, 0.5, txt, 'HorizontalAlignment', 'center', 'FontSize', 12);
    % axis off

    %% COnfusion matrix
clear all;
clc
dataFolder = 'GFD_400';
tarData = {'h_b_30hz_30', 'h_b_30hz_50', 'h_b_30hz_70', 'h_b_30hz_90'};

     load(['../' dataFolder '/' tarData{1}], 'Y');

     T3 = readmatrix("results_EvoN2N.xlsx","Sheet",1,"Range",'E57:M57'); % D28:K28 for PBU, D43:K43 for GFD

     methods = {'SVM', 'DNN', 'DANN', 'DTL','DAFD', 'N2N\_WDA', 'N2N\_DA', 'EvoDCNN', 'EvoN2N'};


    yt = Y.test_results';

    figure
    for i= 1:length(methods)
        subplot(3,3,i)
        yprd = tr2pred(yt, T3(i));
        confusionchart(yt, yprd, "FontSize", 12, 'Title', methods{i}, 'DiagonalColor','#808080');
    end
    % subplot(3,3,9)
    % txt = sprintf('0 = Normal Running  \n1 = Inner Fault (IR) \n2 = Ball Fault (B) \n3 = Outer Race Fault (OR)');
    % txt = sprintf('0 = Healthy/Normal(N)  \n1 = Outer Race (OR) \n2 = Inner Race (IR)');
    % txt = sprintf('0 = Healthy/Normal(N)  \n1 = Broken Tooth (BT)');
    % text(0.5, 0.5, txt, 'HorizontalAlignment', 'center', 'FontSize', 12);
    % axis off
