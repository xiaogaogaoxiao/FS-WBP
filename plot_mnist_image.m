%%
clear;
clc; 
close all;

addpath('data')
addpath('solver')
addpath('subroutine')

ranseed = 1;
rng(ranseed, 'twister');

for digit = 1:9

    %% load data
    maxtime  = 100; 
    imsize   = 56;
    dig_name = strcat('mnist_ss',num2str(digit));

    load(dig_name, '-mat')
    [Imrow, Imcol] = size(images_samesupp(:,:,1));
    scal     = 1; 
    Imrow    = scal*Imrow; 
    Imcol    = scal*Imcol;
    X        = [repmat(1:Imrow, [1, Imcol]); repelem(1:Imcol, 1, Imrow)];   % support points
    Dsingle  = pdist2(X', X', 'sqeuclidean');

    N        = 50;                                      % the sample size for computing a barycenter
    m        = Imrow*Imcol;                              % the size of the centroid
    mt       = Imrow*Imcol*ones(N, 1);                   % mt(i) denotes the size of i-th distribution, i = 1,...,N
    D        = repmat(Dsingle, 1, N);                    % generate D^i
    wt       = zeros(sum(mt), 1);                        % generate w^i
    indpick  = randperm(size(images_samesupp, 3), N);    % pick N images from original dataset

    for i = 1:N
        imgtmp   = images_samesupp(:,:,indpick(i));
        imgblock = mat2cell(imgtmp, (1/scal)*ones(Imrow,1), (1/scal)*ones(Imrow,1));
        imgsmall = cellfun(@sum, cellfun(@sum, imgblock, 'UniformOutput', false)); % rescale image to make the size smaller 
        ind      = sum(mt(1:i-1))+(1:mt(i)); 
        wt(ind)  = imgsmall(:); 
    end

    sample_supp = repmat(X, 1, N);

    input_sparse = 1;
    if input_sparse == 1
        mtnew = zeros(N, 1);
        for i = 1:N
            ind      = sum(mt(1:i-1))+(1:mt(i));
            mtnew(i) = length(find(wt(ind)~=0));
        end
        indnnz      = find(wt~=0);
        wt          = wt(indnnz);
        D           = D(:,indnnz);
        mt          = mtnew;
        sample_supp = sample_supp(:,indnnz);
    end

    %% rescaling. 
    D = D/max(D(:));
    maxT = 500;

    %% IBP
    optsIBP.ibp_vareps     = 0.001;
    optsIBP.ibp_tol        = 1e-10; 
    optsIBP.ibp_max_iters  = 1000000;
    optsIBP.maxtime        = maxtime; 
    optsIBP.display        = 0;
    optsIBP.displayfreq    = 100;
    optsIBP.checkfreq      = 10;
    optsIBP.support_points = X;
    optsIBP.dist_matrix    = D;
    
    [center_ibp, ~, ~] = centroid_IBP(mt, sample_supp, wt', [], optsIBP); 
    
    img_ibp = reshape(center_ibp.w', [Imrow, Imcol]);
    figure; 
    imshow(1-img_ibp, [], 'Colormap', hot, 'InitialMagnification', 400);
    
    ibp_path  = sprintf('figs/%d/ibp_%d', digit, optsIBP.maxtime); 
    saveas(gcf, ibp_path, 'epsc');
    
    %% FIBP. 
    optsFIBP.ibp_vareps     = 0.001;
    optsFIBP.ibp_tol        = 1e-10; 
    optsFIBP.ibp_max_iters  = 1000000;
    optsFIBP.maxtime        = maxtime; 
    optsFIBP.display        = 0;
    optsFIBP.displayfreq    = 100;
    optsFIBP.checkfreq      = 10;
    optsFIBP.support_points = X;
    optsFIBP.dist_matrix    = D;
 
    [center_fibp, ~, ~] = centroid_FIBP(mt, sample_supp, wt', [], optsFIBP);
    
    img_fibp = reshape(center_fibp.w', [Imrow, Imcol]);
    figure; 
    imshow(1-img_fibp, [], 'Colormap', hot, 'InitialMagnification', 400);
    
    fibp_path = sprintf('figs/%d/fibp_%d', digit, optsFIBP.maxtime); 
    saveas(gcf, fibp_path, 'epsc');
end