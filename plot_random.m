%%
clear;
clc; 
close all;

addpath('data')
addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

%% Problem setting: generate D^i, w^i, i = 1,...,N
m = 100;    % the size of the centroid
d = 3;      % dimension of each support point 
num_sample = [20; 50; 100; 200];

ntrials = 10; 

objp_ibp1   = zeros(length(num_sample), 2); 
objp_ibp2   = zeros(length(num_sample), 2);
objp_fibp1  = zeros(length(num_sample), 2); 
objp_fibp2  = zeros(length(num_sample), 2); 
objp_badmm  = zeros(length(num_sample), 2); 

times_gurobi = zeros(length(num_sample), 2);
times_ibp1   = zeros(length(num_sample), 2);
times_ibp2   = zeros(length(num_sample), 2);
times_fibp1  = zeros(length(num_sample), 2);
times_fibp2  = zeros(length(num_sample), 2);
times_badmm  = zeros(length(num_sample), 2); 

for di=1:length(num_sample)
    
    tmp_objp = zeros(ntrials, 5); 
    tmp_time = zeros(ntrials, 6); 
    
    for dj = 1:ntrials
        fprintf('%i\n', dj); 
        N  = num_sample(di);        % N denotes the number of marginals
        mt = m*ones(1, N);          % mt(i) denotes the size of i-th distribution, i = 1,...,N
    
        % generate data
        gm_num       = 5;                         % number of mixtures
        gm_mean      = [-20; -10; 0; 10; 20];     % mean values for the Gaussians means
        sigma        = zeros(1, 1, gm_num); 
        sigma(1,1,:) = 5*ones(gm_num,1);          % variances of the Gaussians
        supp_sample  = [];

        for i = 1:N
            gm_weights  = rand(gm_num, 1); 
            gm_weights  = gm_weights/sum(gm_weights);   % generate the mixture weights
            distrib     = gmdistribution(gm_mean, sigma, gm_weights);
            TMP         = reshape(random(distrib,m*mt(i)), m, mt(i));
            supp_sample = [supp_sample TMP];            % sample the distribution
        end

        wt = zeros(sum(mt), 1); % generate wt = [w1,w2,...,wN]

        for i = 1:N
            ind     = sum(mt(1:i-1))+(1:mt(i)); 
            wt(ind) = rand(mt(i), 1); 
            wt(ind) = wt(ind)/sum(wt(ind));
        end

        censuppmethod = 'kmeans'; % 'random'
        if strcmp(censuppmethod, 'random')
            indpick         = randperm(sum(mt), m);
            supp_center     = supp_sample(:,indpick);
        elseif strcmp(censuppmethod, 'kmeans')
            [~, supp_center] = kmeans(supp_sample', m, 'MaxIter', 10000);
            supp_center      = supp_center';
        end   

        D = pdist2(supp_center', supp_sample', 'sqeuclidean');     % compute D = [D1,D2,...,DN]
        D = max(D, 0);

        weights = rand(1, N); 
        weights = weights/sum(weights);
        Ws = repelem(repmat(weights,m,1),1,mt);
    
        D = Ws.*D;
        D = D/max(D(:));
    
    
        %% call Gurobi
        [Aeq, beq, lb] = generate_coe_matrix(N, m, mt, wt);
        c              = [zeros(m,1); D(:)];   
        model.obj      = c; 
        model.A        = Aeq; 
        model.rhs      = beq; 
        model.lb       = lb;
        model.sense    = '='; 
        model.vtype    = 'C';
    
        params.method     = 1;
        params.OutputFlag = 0;
    
        tic; res_gurobi  = gurobi(model, params); time_gurobi = toc;     
        obj_gurobi = res_gurobi.objval; 
        
        %% call Bregman ADMM
        optsBADMM.badmm_tol       = 1e-4;
        optsBADMM.badmm_max_iters = 5000;
        optsBADMM.display         = 0;
        optsBADMM.displayfreq     = 500;
        optsBADMM.checkfreq       = 200;
        optsBADMM.support_points  = supp_center;
        optsBADMM.dist_matrix     = D;
    
        tic; [~, Pi_badmm] = centroid_BADMM(mt, supp_sample, wt', [], optsBADMM); time_badmm = toc;    
        obj_badmm  = sum(sum(D.*Pi_badmm));
        
        %% call Iterative Bregman Projection
        optsIBP.ibp_vareps      = 0.01;
        optsIBP.ibp_tol         = 1e-6; 
        optsIBP.ibp_max_iters   = 10000;
        optsIBP.display         = 0;
        optsIBP.displayfreq     = 500;
        optsIBP.checkfreq       = 20;
        optsIBP.support_points  = supp_center;
        optsIBP.dist_matrix     = D;
    
        tic; [~, Pi_ibp1] = centroid_IBP(mt, supp_sample, wt', [], optsIBP); time_ibp1 = toc;
        obj_ibp1 = sum(sum(D.*Pi_ibp1));

        optsIBP.ibp_vareps      = 0.001;
        
        tic; [~, Pi_ibp2] = centroid_IBP(mt, supp_sample, wt', [], optsIBP); time_ibp2 = toc;
        obj_ibp2 = sum(sum(D.*Pi_ibp2));
        
        %% call Fast Iterative Bregman Projection
        optsFIBP.ibp_vareps     = 0.01;
        optsFIBP.ibp_tol        = 1e-6; 
        optsFIBP.ibp_max_iters  = 1000000;
        optsFIBP.display        = 0;
        optsFIBP.displayfreq    = 500;
        optsFIBP.checkfreq      = 200;
        optsFIBP.support_points = supp_center;
        optsFIBP.dist_matrix    = D;
        
        tic; [~, Pi_fibp1] = centroid_FIBP(mt, supp_sample, wt', [], optsFIBP); time_fibp1 = toc;
        obj_fibp1 = sum(sum(D.*Pi_fibp1));
        
        optsFIBP.ibp_vareps = 0.001;
    
        tic; [~, Pi_fibp2] = centroid_FIBP(mt, supp_sample, wt', [], optsFIBP); time_fibp2 = toc;
        obj_fibp2 = sum(sum(D.*Pi_fibp2));
        
        %% set the result at each round
        tmp_objp(dj, 1) = abs(obj_ibp1-obj_gurobi)/abs(obj_gurobi);
        tmp_objp(dj, 2) = abs(obj_ibp2-obj_gurobi)/abs(obj_gurobi);
        tmp_objp(dj, 3) = abs(obj_fibp1-obj_gurobi)/abs(obj_gurobi);
        tmp_objp(dj, 4) = abs(obj_fibp2-obj_gurobi)/abs(obj_gurobi);
        tmp_objp(dj, 5) = abs(obj_badmm-obj_gurobi)/abs(obj_gurobi);
        
        tmp_time(dj, 1) = time_gurobi;
        tmp_time(dj, 2) = time_ibp1; 
        tmp_time(dj, 3) = time_ibp2; 
        tmp_time(dj, 4) = time_fibp1; 
        tmp_time(dj, 5) = time_fibp2; 
        tmp_time(dj, 6) = time_badmm; 
    end
    objp_ibp1(di, 1) = mean(tmp_objp(:, 1)); 
    objp_ibp1(di, 2) = std(tmp_objp(:, 1)); 
    objp_ibp2(di, 1) = mean(tmp_objp(:, 2)); 
    objp_ibp2(di, 2) = std(tmp_objp(:, 2)); 
    objp_fibp1(di, 1) = mean(tmp_objp(:, 3)); 
    objp_fibp1(di, 2) = std(tmp_objp(:, 3));
    objp_fibp2(di, 1) = mean(tmp_objp(:, 4)); 
    objp_fibp2(di, 2) = std(tmp_objp(:, 4)); 
    objp_badmm(di, 1) = mean(tmp_objp(:, 5)); 
    objp_badmm(di, 2) = std(tmp_objp(:, 5)); 
    
    times_gurobi(di, 1) = mean(tmp_time(:, 1)); 
    times_gurobi(di, 2) = std(tmp_time(:, 1)); 
    times_ibp1(di, 1) = mean(tmp_time(:, 2)); 
    times_ibp1(di, 2) = std(tmp_time(:, 2));
    times_ibp2(di, 1) = mean(tmp_time(:, 3)); 
    times_ibp2(di, 2) = std(tmp_time(:, 3));
    times_fibp1(di, 1) = mean(tmp_time(:, 4)); 
    times_fibp1(di, 2) = std(tmp_time(:, 4));
    times_fibp2(di, 1) = mean(tmp_time(:, 5)); 
    times_fibp2(di, 2) = std(tmp_time(:, 5));
    times_badmm(di, 1) = mean(tmp_time(:, 6)); 
    times_badmm(di, 2) = std(tmp_time(:, 6));
end

figure; 
errorbar(num_sample, objp_fibp2(:, 1), objp_fibp2(:, 2), '-d', 'LineWidth', 3, 'MarkerSize', 10);
hold on
errorbar(num_sample, objp_fibp1(:, 1), objp_fibp1(:, 2), '-*', 'LineWidth', 3, 'MarkerSize', 10);
errorbar(num_sample, objp_ibp2(:, 1), objp_ibp2(:, 2), '-o', 'LineWidth', 3, 'MarkerSize', 10);
errorbar(num_sample, objp_ibp1(:, 1), objp_ibp1(:, 2), '-s', 'LineWidth', 3, 'MarkerSize', 10);
errorbar(num_sample, objp_badmm(:, 1), objp_badmm(:, 2), '-x', 'LineWidth', 3, 'MarkerSize', 10);
hold off
legend('f2', 'f1', 'i2', 'i1', 'b', 'Location', 'northwest', 'Orientation','horizontal');

set(gca, 'YScale','log');
set(gca, 'FontSize', 20);
xlabel('numer of marginals (m)');
ylabel('normalized obj');
xlim([min(num_sample) max(num_sample)])
ylim([0.001 5])
title(['n=', num2str(m)]);

path = sprintf('figs/random_obj_%d', m); 
saveas(gcf, path, 'epsc');


figure; 
errorbar(num_sample, times_fibp2(:, 1), times_fibp2(:, 2), '-d', 'LineWidth', 3, 'MarkerSize', 10);
hold on
errorbar(num_sample, times_fibp1(:, 1), times_fibp1(:, 2), '-*', 'LineWidth', 3, 'MarkerSize', 10);
errorbar(num_sample, times_ibp2(:, 1), times_ibp2(:, 2), '-o', 'LineWidth', 3, 'MarkerSize', 10);
errorbar(num_sample, times_ibp1(:, 1), times_ibp1(:, 2), '-s', 'LineWidth', 3, 'MarkerSize', 10);
errorbar(num_sample, times_badmm(:, 1), times_badmm(:, 2), '-x', 'LineWidth', 3, 'MarkerSize', 10);
errorbar(num_sample, times_gurobi(:, 1), times_gurobi(:, 2), '-+', 'LineWidth', 3, 'MarkerSize', 10);
hold off
legend('f2', 'f1', 'i2', 'i1', 'b', 'g', 'Location', 'northwest', 'Orientation','horizontal');

set(gca, 'YScale','log');
set(gca, 'FontSize', 20);
xlabel('numer of marginals (m)');
ylabel('time (in seconds)');
ylim([0 1000])
xlim([min(num_sample) max(num_sample)])
title(['n=', num2str(m)]);

path = sprintf('figs/random_time_%d', m); 
saveas(gcf, path, 'epsc');
