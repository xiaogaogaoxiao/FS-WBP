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
num_sample = [200; 500; 1000; 2000];

ntrials = 5; 
times_gurobi = zeros(length(num_sample), 2);
times_fibp = zeros(length(num_sample), 2);

for di=1:length(num_sample)
    
    tmp_objp = zeros(ntrials, 1); 
    tmp_feas = zeros(ntrials, 2); 
    tmp_iter = zeros(ntrials, 2); 
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
    
        w_gurobi  = res_gurobi.x(1:m); 
        Pi_gurobi = zeros(m, sum(mt));
    
        for i = 1:N
            ind   = sum(mt(1:i-1))+(1:mt(i)); 
            ind1  = m+m*sum(mt(1:i-1))+1;  
            pitmp = res_gurobi.x(ind1:ind1+m*mt(i)-1);
            Pi_gurobi(:,ind) = reshape(pitmp, [m, mt(i)]);
        end
    
        feas_gurobi     = compfeas(Pi_gurobi, w_gurobi, wt, N, mt);    
        obj_gurobi      = res_gurobi.objval; 
        gap_gurobi      = abs(res_gurobi.objval-res_gurobi.pi'*beq)/(1+abs(res_gurobi.objval)+abs(res_gurobi.pi'*beq));
        feas_gurobi_new = max(norm(Aeq*res_gurobi.x-beq), norm(min(res_gurobi.x,0)))/(1+norm(res_gurobi.x));
        KKTres_gurobi   = max([feas_gurobi_new, norm(max(Aeq'*res_gurobi.pi-c,0))/(1+norm(res_gurobi.pi)), ...
            norm(res_gurobi.x'*(c-Aeq'*res_gurobi.pi))/(1+norm(res_gurobi.x)+norm(c-Aeq'*res_gurobi.pi))]);
    

        %% call Fast Iterative Bregman Projection
        optsFIBP.ibp_vareps     = 0.001;
        optsFIBP.ibp_tol        = 1e-6; 
        optsFIBP.ibp_max_iters  = 1000000;
        optsFIBP.display        = 0;
        optsFIBP.displayfreq    = 500;
        optsFIBP.checkfreq      = 200;
        optsFIBP.support_points = supp_center;
        optsFIBP.dist_matrix    = D;
        
        tic; 
        [center_fibp, Pi_fibp, iter_fibp] = centroid_FIBP(mt, supp_sample, wt', [], optsFIBP); 
        time_fibp = toc;
    
        w_fibp    = center_fibp.w';
        obj_fibp  = sum(sum(D.*Pi_fibp));
        feas_fibp = compfeas(Pi_fibp, w_fibp, wt, N, mt);
        
        tmp_objp(dj)    = abs(obj_fibp-obj_gurobi)/abs(obj_gurobi); 
        tmp_feas(dj, 1) = feas_gurobi; 
        tmp_feas(dj, 2) = feas_fibp; 
        tmp_iter(dj, 1) = res_gurobi.itercount;
        tmp_iter(dj, 2) = iter_fibp; 
        tmp_time(dj, 1) = time_gurobi;
        tmp_time(dj, 2) = time_fibp; 
    end
    
    times_gurobi(di, 1) = mean(tmp_time(:, 1)); 
    times_gurobi(di, 2) = std(tmp_time(:, 1)); 
    
    times_fibp(di, 1) = mean(tmp_time(:, 2)); 
    times_fibp(di, 2) = std(tmp_time(:, 2));
    
    %% print results
    fprintf('\n'); 
    fprintf('method    | g     | f2\n');
    fprintf('obj  & -             & %0.2e (%0.2e) \n', mean(tmp_objp), std(tmp_objp)); 
    fprintf('feas & %0.2e (%0.2e) & %0.2e (%0.2e) \n', mean(tmp_feas(:, 1)), std(tmp_feas(:, 1)), ...
        mean(tmp_feas(:, 2)), std(tmp_feas(:, 2)));
    fprintf('iter & %i (%i)       & %i (%i) \n', round(mean(tmp_iter(:, 1))), round(std(tmp_iter(:, 1))), ...
        round(mean(tmp_iter(:, 2))), round(std(tmp_iter(:, 2))));
    fprintf('time & %5.2f (%5.2f) & %5.2f (%5.2f) \n', times_gurobi(di, 1), times_gurobi(di, 2), ...
        times_fibp(di, 1), times_fibp(di, 2));
end

figure; 
errorbar(num_sample, times_fibp(:, 1), times_fibp(:, 2)/2, '-s', 'LineWidth', 4, 'MarkerSize', 12); 
hold on
errorbar(num_sample, times_gurobi(:, 1), times_gurobi(:, 2), '-d', 'LineWidth', 4, 'MarkerSize', 12); 
hold off
legend('f2', 'g', 'Location', 'northwest');

set(gca, 'YScale','log');
set(gca, 'FontSize', 20);
xlabel('numer of marginals (m)');
ylabel('time (in seconds)');
xlim([min(num_sample) max(num_sample)])
title(['n=', num2str(m)]);

path = sprintf('figs/gurobi_%d', m); 
saveas(gcf, path, 'epsc');





