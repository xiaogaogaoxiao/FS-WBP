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
num_sample = [20 50 100 200];

ntrials = 10; 

objp_proxibp   = zeros(length(num_sample), 2);
objp_proxfibp  = zeros(length(num_sample), 2); 

times_proxibp   = zeros(length(num_sample), 2);
times_proxfibp  = zeros(length(num_sample), 2);

for di=1:length(num_sample)
    
    tmp_objp = zeros(ntrials, 2); 
    tmp_time = zeros(ntrials, 2); 
    
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
    
        tic; 
        res_gurobi  = gurobi(model, params); 
        time_gurobi = toc;     
        obj_gurobi = res_gurobi.objval; 
        
        %% call Proximal Iterative Bregman Projection 
        optsIBP.ibp_vareps      = 0.01;
        optsIBP.ibp_tol         = 1e-2;
        optsIBP.ibp_max_iters   = 10000;
        optsIBP.display         = 0;
        optsIBP.displayfreq     = 500;
        optsIBP.checkfreq       = 20;
        optsIBP.support_points  = supp_center;
        optsIBP.dist_matrix     = D;
    
        tic; [Pi_proxibp, ~] = centroid_proxIBP(mt, supp_sample, wt', [], optsIBP); time_proxibp = toc;
        obj_proxibp = sum(sum(D.*Pi_proxibp));
        
        %% call Fast Proximal Iterative Bregman Projection
        optsFIBP.ibp_vareps     = 0.1;
        optsFIBP.ibp_tol        = 1e-2; 
        optsFIBP.ibp_max_iters  = 1000000;
        optsFIBP.display        = 0;
        optsFIBP.displayfreq    = 500;
        optsFIBP.checkfreq      = 200;
        optsFIBP.support_points = supp_center;
        optsFIBP.dist_matrix    = D;
        
        tic; [Pi_proxfibp, ~] = centroid_proxFIBP(mt, supp_sample, wt', [], optsFIBP); time_proxfibp = toc;
        obj_proxfibp = sum(sum(D.*Pi_proxfibp));
        
        %% set the result at each round
        tmp_objp(dj, 1) = abs(obj_proxibp-obj_gurobi)/abs(obj_gurobi);
        tmp_objp(dj, 2) = abs(obj_proxfibp-obj_gurobi)/abs(obj_gurobi);
        
        tmp_time(dj, 1) = time_proxibp; 
        tmp_time(dj, 2) = time_proxfibp; 
    end
    objp_proxibp(di, 1) = mean(tmp_objp(:, 1)); 
    objp_proxibp(di, 2) = std(tmp_objp(:, 1)); 
    objp_proxfibp(di, 1) = mean(tmp_objp(:, 2)); 
    objp_proxfibp(di, 2) = std(tmp_objp(:, 2)); 
    
    times_proxibp(di, 1) = mean(tmp_time(:, 1)); 
    times_proxibp(di, 2) = std(tmp_time(:, 1));
    times_proxfibp(di, 1) = mean(tmp_time(:, 2)); 
    times_proxfibp(di, 2) = std(tmp_time(:, 2));
end

figure; 
errorbar(num_sample, objp_proxfibp(:, 1), objp_proxfibp(:, 2), '-d', 'LineWidth', 3, 'MarkerSize', 10);
hold on
errorbar(num_sample, objp_proxibp(:, 1), objp_proxibp(:, 2), '-o', 'LineWidth', 3, 'MarkerSize', 10);
hold off
legend('f (prox)', 'i (prox)', 'Location', 'northwest', 'Orientation','horizontal');

set(gca, 'YScale','log');
set(gca, 'FontSize', 15);
xlabel('numer of marginals (m)');
ylabel('normalized obj');
xlim([min(num_sample) max(num_sample)])
title(['n=', num2str(m)]);

path = sprintf('figs/prox_obj_%d', m); 
saveas(gcf, path, 'epsc');


figure; 
errorbar(num_sample, times_proxfibp(:, 1), times_proxfibp(:, 2), '-d', 'LineWidth', 3, 'MarkerSize', 10);
hold on
errorbar(num_sample, times_proxibp(:, 1), times_proxibp(:, 2), '-o', 'LineWidth', 3, 'MarkerSize', 10);
hold off
legend('f (prox)', 'i (prox)', 'Location', 'northwest', 'Orientation','horizontal');

set(gca, 'YScale','log');
set(gca, 'FontSize', 15);
xlabel('numer of marginals (m)');
ylabel('time (in seconds)');
xlim([min(num_sample) max(num_sample)])
title(['n=', num2str(m)]);

path = sprintf('figs/prox_time_%d', m); 
saveas(gcf, path, 'epsc');
