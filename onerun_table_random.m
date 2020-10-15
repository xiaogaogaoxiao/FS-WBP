%%************************************************************************* 
%% Call different algorithms to compute Wasserstein barycenters
%% minimize      <D^1, P^1> + ... + <D^N, P^N>
%% subject to    P^i * e^i = w,
%%               (P^i)^T * e = w^i,
%%               P^i >= 0, P^i \in R^{m*m_i}, i = 1,...,N
%%               e^T * w = 1, w >=0, w \in R^{m}.
%%*************************************************************************

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
d = 3;      % dimension of each support point 
num_sample = [20; 50; 100; 200];
num_centroid = [50; 100; 200]; 

ntrials = 10; 

objp_ibp1   = zeros(length(num_sample), length(num_centroid), 2); 
objp_ibp2   = zeros(length(num_sample), length(num_centroid), 2);
objp_fibp1  = zeros(length(num_sample), length(num_centroid), 2); 
objp_fibp2  = zeros(length(num_sample), length(num_centroid), 2); 
objp_badmm  = zeros(length(num_sample), length(num_centroid), 2); 

infeas_gurobi = zeros(length(num_sample), length(num_centroid), 2);
infeas_ibp1   = zeros(length(num_sample), length(num_centroid), 2); 
infeas_ibp2   = zeros(length(num_sample), length(num_centroid), 2);
infeas_fibp1  = zeros(length(num_sample), length(num_centroid), 2); 
infeas_fibp2  = zeros(length(num_sample), length(num_centroid), 2); 
infeas_badmm  = zeros(length(num_sample), length(num_centroid), 2); 

iters_gurobi = zeros(length(num_sample), length(num_centroid), 2);
iters_ibp1   = zeros(length(num_sample), length(num_centroid), 2);
iters_ibp2   = zeros(length(num_sample), length(num_centroid), 2);
iters_fibp1  = zeros(length(num_sample), length(num_centroid), 2);
iters_fibp2  = zeros(length(num_sample), length(num_centroid), 2);
iters_badmm  = zeros(length(num_sample), length(num_centroid), 2); 

times_gurobi = zeros(length(num_sample), length(num_centroid), 2);
times_ibp1   = zeros(length(num_sample), length(num_centroid), 2);
times_ibp2   = zeros(length(num_sample), length(num_centroid), 2);
times_fibp1  = zeros(length(num_sample), length(num_centroid), 2);
times_fibp2  = zeros(length(num_sample), length(num_centroid), 2);
times_badmm  = zeros(length(num_sample), length(num_centroid), 2); 

for di=1:length(num_sample)
    for dj=1:length(num_centroid)
        
        tmp_objp   = zeros(ntrials, 5); 
        tmp_infeas = zeros(ntrials, 6);
        tmp_iter   = zeros(ntrials, 6);
        tmp_time   = zeros(ntrials, 6);
        
        for dn = 1:ntrials
            
            fprintf('%i\t', dn); 
            N = num_sample(di);        % N denotes the number of marginals
            m = num_centroid(dj);      % m denotes the number of support points 
            mt = m*ones(1, N);         % mt(i) denotes the size of i-th distribution, i = 1,...,N
            
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
    
            tic; res_gurobi = gurobi(model, params); time_gurobi = toc;     
            w_gurobi = res_gurobi.x(1:m); 
            Pi_gurobi = zeros(m, sum(mt));
            
            for i = 1:N
                ind     = sum(mt(1:i-1))+(1:mt(i)); 
                ind1    = m+m*sum(mt(1:i-1))+1;  
                pitmp   = res_gurobi.x(ind1:ind1+m*mt(i)-1);
                Pi_gurobi(:,ind) = reshape(pitmp, [m, mt(i)]);
            end
            
            obj_gurobi = res_gurobi.objval;
            infeas_gurobi = compfeas(Pi_gurobi, w_gurobi, wt, N, mt); 
            iter_gurobi = res_gurobi.itercount;
            
            %% call Bregman ADMM
            optsBADMM.badmm_tol       = 1e-4;
            optsBADMM.badmm_max_iters = 5000;
            optsBADMM.display         = 0;
            optsBADMM.displayfreq     = 500;
            optsBADMM.checkfreq       = 200;
            optsBADMM.support_points  = supp_center;
            optsBADMM.dist_matrix     = D;
    
            tic; [center_badmm, Pi_badmm, iter_badmm] = centroid_BADMM(mt, supp_sample, wt', [], optsBADMM); time_badmm = toc;
            w_badmm = center_badmm.w';
            
            obj_badmm = sum(sum(D.*Pi_badmm));
            infeas_badmm = compfeas(Pi_badmm, w_badmm, wt, N, mt);
            
            %% call Iterative Bregman Projection
            optsIBP.ibp_vareps      = 0.01;
            optsIBP.ibp_tol         = 1e-6; 
            optsIBP.ibp_max_iters   = 1000000;
            optsIBP.display         = 0;
            optsIBP.displayfreq     = 500;
            optsIBP.checkfreq       = 200;
            optsIBP.support_points  = supp_center;
            optsIBP.dist_matrix     = D;
    
            tic; [center_ibp1, Pi_ibp1, iter_ibp1] = centroid_IBP(mt, supp_sample, wt', [], optsIBP); time_ibp1 = toc;
            w_ibp1 = center_ibp1.w';
            
            optsIBP.ibp_vareps      = 0.001;
            tic; [center_ibp2, Pi_ibp2, iter_ibp2] = centroid_IBP(mt, supp_sample, wt', [], optsIBP); time_ibp2 = toc;
            w_ibp2 = center_ibp2.w';
            
            obj_ibp1 = sum(sum(D.*Pi_ibp1));
            obj_ibp2 = sum(sum(D.*Pi_ibp2));
            infeas_ibp1 = compfeas(Pi_ibp1, w_ibp1, wt, N, mt);
            infeas_ibp2 = compfeas(Pi_ibp2, w_ibp2, wt, N, mt);
            
            %% call Fast Iterative Bregman Projection
            optsFIBP.ibp_vareps     = 0.01;
            optsFIBP.ibp_tol        = 1e-6; 
            optsFIBP.ibp_max_iters  = 1000000;
            optsFIBP.display        = 0;
            optsFIBP.displayfreq    = 500;
            optsFIBP.checkfreq      = 200;
            optsFIBP.support_points = supp_center;
            optsFIBP.dist_matrix    = D;
        
            tic; [center_fibp1, Pi_fibp1, iter_fibp1] = centroid_FIBP(mt, supp_sample, wt', [], optsFIBP); time_fibp1 = toc;
            w_fibp1 = center_fibp1.w';
            
            optsFIBP.ibp_vareps     = 0.001;
            tic; [center_fibp2, Pi_fibp2, iter_fibp2] = centroid_FIBP(mt, supp_sample, wt', [], optsFIBP); time_fibp2 = toc;
            w_fibp2 = center_fibp2.w';
            
            obj_fibp1 = sum(sum(D.*Pi_fibp1));
            obj_fibp2 = sum(sum(D.*Pi_fibp2));
            infeas_fibp1 = compfeas(Pi_fibp1, w_fibp1, wt, N, mt);
            infeas_fibp2 = compfeas(Pi_fibp2, w_fibp2, wt, N, mt);
            
            %% set the result at each round
            tmp_objp(dn, 1) = abs(obj_ibp1-obj_gurobi)/abs(obj_gurobi);
            tmp_objp(dn, 2) = abs(obj_ibp2-obj_gurobi)/abs(obj_gurobi);
            tmp_objp(dn, 3) = abs(obj_fibp1-obj_gurobi)/abs(obj_gurobi);
            tmp_objp(dn, 4) = abs(obj_fibp2-obj_gurobi)/abs(obj_gurobi);
            tmp_objp(dn, 5) = abs(obj_badmm-obj_gurobi)/abs(obj_gurobi);
            
            tmp_infeas(dn, 1) = infeas_gurobi;
            tmp_infeas(dn, 2) = infeas_ibp1; 
            tmp_infeas(dn, 3) = infeas_ibp2; 
            tmp_infeas(dn, 4) = infeas_fibp1; 
            tmp_infeas(dn, 5) = infeas_fibp2; 
            tmp_infeas(dn, 6) = infeas_badmm; 
            
            tmp_iter(dn, 1) = iter_gurobi;
            tmp_iter(dn, 2) = iter_ibp1; 
            tmp_iter(dn, 3) = iter_ibp2; 
            tmp_iter(dn, 4) = iter_fibp1; 
            tmp_iter(dn, 5) = iter_fibp2; 
            tmp_iter(dn, 6) = iter_badmm; 
        
            tmp_time(dn, 1) = time_gurobi;
            tmp_time(dn, 2) = time_ibp1; 
            tmp_time(dn, 3) = time_ibp2; 
            tmp_time(dn, 4) = time_fibp1; 
            tmp_time(dn, 5) = time_fibp2; 
            tmp_time(dn, 6) = time_badmm; 
        end
        
        fprintf('\n');
        objp_ibp1(di, dj, 1) = mean(tmp_objp(:, 1));
        objp_ibp1(di, dj, 2) = std(tmp_objp(:, 1)); 
        objp_ibp2(di, dj, 1) = mean(tmp_objp(:, 2)); 
        objp_ibp2(di, dj, 2) = std(tmp_objp(:, 2)); 
        objp_fibp1(di, dj, 1) = mean(tmp_objp(:, 3)); 
        objp_fibp1(di, dj, 2) = std(tmp_objp(:, 3));
        objp_fibp2(di, dj, 1) = mean(tmp_objp(:, 4)); 
        objp_fibp2(di, dj, 2) = std(tmp_objp(:, 4)); 
        objp_badmm(di, dj, 1) = mean(tmp_objp(:, 5)); 
        objp_badmm(di, dj, 2) = std(tmp_objp(:, 5)); 
        
        infeas_gurobi(di, dj, 1) = mean(tmp_infeas(:, 1)); 
        infeas_gurobi(di, dj, 2) = std(tmp_infeas(:, 1)); 
        infeas_ibp1(di, dj, 1) = mean(tmp_infeas(:, 2)); 
        infeas_ibp1(di, dj, 2) = std(tmp_infeas(:, 2));
        infeas_ibp2(di, dj, 1) = mean(tmp_infeas(:, 3)); 
        infeas_ibp2(di, dj, 2) = std(tmp_infeas(:, 3));
        infeas_fibp1(di, dj, 1) = mean(tmp_infeas(:, 4)); 
        infeas_fibp1(di, dj, 2) = std(tmp_infeas(:, 4));
        infeas_fibp2(di, dj, 1) = mean(tmp_infeas(:, 5)); 
        infeas_fibp2(di, dj, 2) = std(tmp_infeas(:, 5));
        infeas_badmm(di, dj, 1) = mean(tmp_infeas(:, 6)); 
        infeas_badmm(di, dj, 2) = std(tmp_infeas(:, 6));
        
        iters_gurobi(di, dj, 1) = mean(tmp_iter(:, 1)); 
        iters_gurobi(di, dj, 2) = std(tmp_iter(:, 1)); 
        iters_ibp1(di, dj, 1) = mean(tmp_iter(:, 2)); 
        iters_ibp1(di, dj, 2) = std(tmp_iter(:, 2));
        iters_ibp2(di, dj, 1) = mean(tmp_iter(:, 3)); 
        iters_ibp2(di, dj, 2) = std(tmp_iter(:, 3));
        iters_fibp1(di, dj, 1) = mean(tmp_iter(:, 4)); 
        iters_fibp1(di, dj, 2) = std(tmp_iter(:, 4));
        iters_fibp2(di, dj, 1) = mean(tmp_iter(:, 5)); 
        iters_fibp2(di, dj, 2) = std(tmp_iter(:, 5));
        iters_badmm(di, dj, 1) = mean(tmp_iter(:, 6)); 
        iters_badmm(di, dj, 2) = std(tmp_iter(:, 6));
        
        times_gurobi(di, dj, 1) = mean(tmp_time(:, 1)); 
        times_gurobi(di, dj, 2) = std(tmp_time(:, 1)); 
        times_ibp1(di, dj, 1) = mean(tmp_time(:, 2)); 
        times_ibp1(di, dj, 2) = std(tmp_time(:, 2));
        times_ibp2(di, dj, 1) = mean(tmp_time(:, 3)); 
        times_ibp2(di, dj, 2) = std(tmp_time(:, 3));
        times_fibp1(di, dj, 1) = mean(tmp_time(:, 4)); 
        times_fibp1(di, dj, 2) = std(tmp_time(:, 4));
        times_fibp2(di, dj, 1) = mean(tmp_time(:, 5)); 
        times_fibp2(di, dj, 2) = std(tmp_time(:, 5));
        times_badmm(di, dj, 1) = mean(tmp_time(:, 6)); 
        times_badmm(di, dj, 2) = std(tmp_time(:, 6));
    end
end
   
%% print results
fprintf('\n'); 

fprintf('normalized objective value\n'); 
for di=1:length(num_sample)
    for dj=1:length(num_centroid) 
        N = num_sample(di);        
        m = num_centroid(dj);      
        fprintf('%i & %i & - & %0.1e (%0.1e) & %0.1e (%0.1e) & %0.1e (%0.1e) & %0.1e (%0.1e) & %0.1e (%0.1e) \n', ...
            N, m, objp_badmm(di, dj, 1), objp_badmm(di, dj, 2), ...
            objp_ibp1(di, dj, 1), objp_ibp1(di, dj, 2), objp_ibp2(di, dj, 1), objp_ibp2(di, dj, 2), ...
            objp_fibp1(di, dj, 1), objp_fibp1(di, dj, 2), objp_fibp2(di, dj, 1), objp_fibp2(di, dj, 2));  
    end
end

fprintf('infeasibility violation\n');
for di=1:length(num_sample)
    for dj=1:length(num_centroid) 
        N = num_sample(di);        
        m = num_centroid(dj);      
        fprintf('%i & %i & %0.2e (%0.2e) & %0.2e (%0.2e) & %0.2e (%0.2e) & %0.2e (%0.2e) & %0.2e (%0.2e) & %0.2e (%0.2e) \n', ...
            N, m, infeas_gurobi(di, dj, 1), infeas_gurobi(di, dj, 2), infeas_badmm(di, dj, 1), infeas_badmm(di, dj, 2), ...
            infeas_ibp1(di, dj, 1), infeas_ibp1(di, dj, 2), infeas_ibp2(di, dj, 1), infeas_ibp2(di, dj, 2), ...
            infeas_fibp1(di, dj, 1), infeas_fibp1(di, dj, 2), infeas_fibp2(di, dj, 1), infeas_fibp2(di, dj, 2));  
    end
end

fprintf('iteration count\n');
for di=1:length(num_sample)
    for dj=1:length(num_centroid) 
        N = num_sample(di);        
        m = num_centroid(dj);      
        fprintf('%i & %i & %i (%i) & %i (%i) & %i (%i) & %i (%i) & %i (%i) & %i (%i) \n', ...
            N, m, iters_gurobi(di, dj, 1), iters_gurobi(di, dj, 2), iters_badmm(di, dj, 1), iters_badmm(di, dj, 2), ...
            iters_ibp1(di, dj, 1), iters_ibp1(di, dj, 2), iters_ibp2(di, dj, 1), iters_ibp2(di, dj, 2), ...
            iters_fibp1(di, dj, 1), iters_fibp1(di, dj, 2), iters_fibp2(di, dj, 1), iters_fibp2(di, dj, 2));  
    end
end

fprintf('computational time\n');
for di=1:length(num_sample)
    for dj=1:length(num_centroid) 
        N = num_sample(di);        
        m = num_centroid(dj);      
        fprintf('%i & %i & %0.1e (%0.1e) & %0.1e (%0.1e) & %0.1e (%0.1e) & %0.1e (%0.1e) & %0.1e (%0.1e) & %0.1e (%0.1e) \n', ...
            N, m, times_gurobi(di, dj, 1), times_gurobi(di, dj, 2), times_badmm(di, dj, 1), times_badmm(di, dj, 2), ...
            times_ibp1(di, dj, 1), times_ibp1(di, dj, 2), times_ibp2(di, dj, 1), times_ibp2(di, dj, 2), ...
            times_fibp1(di, dj, 1), times_fibp1(di, dj, 2), times_fibp2(di, dj, 1), times_fibp2(di, dj, 2));  
    end
end



