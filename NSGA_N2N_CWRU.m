clear all; clc
addpath ('./ReqFnNSGAII'); addpath ('./ReqFnNSGAII/softmax')
addpath ./ReqFnNSGAII/minFunc/;

Np      = 100;       % Number of chromosomes in the population
maxgen  = 20;       % Maximum number of generations
pc      = 0.8;       % Probability of crossover
pm      = 0.2;       % Probability of mutation
nVar    = 2;         % Number of variables (dimensions or objectives)
var_min = [1, 8];    % Minimum value for each gen
var_max = [6,90];    % Maximum value for each gen

para.mini_batch_size =1;      % mini bacth size
para.eta=0.9;
para.momentum = 0.9;
para.epochs_fineTune=20;

%% load teacher model and training data

dataFolder = 'CWRU_400';
srcData  = 'src_7_0';
tarData = {'FE_tar_7_1', 'FE_tar_7_2', 'FE_tar_7_3', 'FE_tar_14_1', 'FE_tar_14_2',...
    'FE_tar_14_3', 'FE_tar_21_1', 'FE_tar_21_2', 'FE_tar_21_3', 'ims_tar'};

% results = struct('gen', cell(1,maxgen+1), 'best_net', cell(1,maxgen+1), 'MSE', cell(1,maxgen+1));
acc_log = zeros(maxgen+1,1);
fileID = fopen('./logs_CWRU/NSGA_net2net_cwru400.txt','w');
%% load dataset
fprintf(fileID, '\n\n*****Target data: 400 samples/Class\n');

for ca=1:length(tarData)
    
%     Y = train_test_val(dataFolder, tarData{ca});
    load(['./' dataFolder '/' tarData{ca}], 'Y');
    
    
    %% ==========Initialization====================
    gen = 0;
    P = genPop(Np, var_min, var_max);   % Np number of different solution
    load('./logs_CWRU/TeacherNet', 'net')
    P{1} = [length(net.nh) net.nh]; % Replace 1st chromosome with the teacher architecture
    [Pfit, P, best_net]  = net2net(gen, net, P, Y, para); % Accuracy of selected network architecture
    fprintf('Gen: #%d, \t best model: Acc=%f\n\n', gen, best_net.ACC);
    acc_log(1) = best_net.ACC;
    %         results(1).gen = gen; results(1).best_net=best_net; results(1).acc = best_net.ACC;
    Prank = FastNonDominatedSorting_Vectorized(Pfit);
    [P,~] = selectParentByRank(P, Prank);
    Q = applyCrossoverAndMutation(P, pc, pm, var_max, var_min);
    
    
    %% ========================================================
    % Main NSGA-II loop (evolve through generations)
    for gen = 1:maxgen
        % (i) Merge the parent and the children
        R = [P; Q];
        
        % (ii) Compute the new fitness and Pareto Fronts
        [Rfit, R, best_net]  = net2net(gen, best_net, R, Y, para);
        Rrank = FastNonDominatedSorting_Vectorized(Rfit);
        fprintf('Gen: #%d:  FF size: %d,  best model: ACC=%f\n', gen, sum(Rrank==1), best_net.ACC);
        
        % (iv) Sort by rank
        [Rrank,idx] = sort(Rrank,'ascend');
        Rfit = Rfit(idx,:);
        R = R(idx,:);
        
        % (v) Compute the crowding distance index
        [Rcrowd, Rrank,~,R] = crowdingDistances(Rrank, Rfit, R);
        
        % (vi) Select Parent
        P = selectParentByRankAndDistance(Rcrowd, Rrank, R);
        
        % (vii) Compute child
        Q = applyCrossoverAndMutation(P, pc, pm, var_max, var_min);
        %             results(gen+1).gen = gen; results(gen+1).best_net=best_net;
        %             results(gen+1).acc = best_net.ACC;
        acc_log(gen+1) = best_net.ACC;
    end
    save(['./logs_CWRU/Out_' tarData{ca} '_400'], 'best_net', 'acc_log')
    
    xt = Y.test_inputs';
    yt = Y.test_results';
    if min(yt)==0
        yt = yt+1;
    end
    
    [acc,~, hData, ~] = valNetSfC(xt, yt, best_net);
    acc_mat(ca,1) = acc*100;
    fprintf(fileID, 'Accuracy of best model for %s: %.2f\n', tarData{ca}, acc*100);
    
    
end
save('./logs_CWRU/NSGA_net2net_400.mat', 'acc_mat', 'hData')
fclose(fileID);
%%
rmpath ('./ReqFnNSGAII'); rmpath ('./ReqFnNSGAII/softmax')
rmpath ./ReqFnNSGAII/minFunc/;
