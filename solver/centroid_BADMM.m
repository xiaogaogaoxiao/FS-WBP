%%************************************************************************
%% Call Bregman ADMM to compute Wasserstein Barycenter
function [c, X, iter, whist, timehist] = centroid_BADMM(stride, supp, w, c0, options)
% The algorithmic prototype of Wasserstein Barycenter using Bregman ADMM

if size(stride,1)~=1, stride = stride'; end
if size(w,1)~=1, w = w'; end

n = length(stride);
m = length(w);
posvec = [1, cumsum(stride)+1];

if isempty(c0)
    c = centroid_init(stride, supp, w, options);
else
    c = c0;
end

support_size = length(c.w);

X = zeros(support_size, m);
Y = zeros(size(X)); 
Z = X;

spIDX_rows = zeros(support_size*m, 1);
spIDX_cols = zeros(support_size*m, 1);

for i = 1 : n
    [xx, yy] = meshgrid((i-1)*support_size + (1:support_size), posvec(i):posvec(i+1)-1);
    ii = support_size*(posvec(i)-1) + (1:(support_size*stride(i)));
    spIDX_rows(ii) = xx';
    spIDX_cols(ii) = yy';
end

spIDX = repmat(speye(support_size), [1, n]);

% initialization
for i = 1 : n
    Z(:,posvec(i):posvec(i+1)-1) = 1/(support_size*stride(i));
end

if isfield(options, 'dist_matrix')
    C = options.dist_matrix;
else
    C = pdist2(c.supp', supp', 'sqeuclidean');
end

nIter = 20000;
if isfield(options, 'badmm_max_iters')
    nIter = options.badmm_max_iters;
end

if isfield(options, 'badmm_rho')
    rho = options.badmm_rho;   % compute rho in advance to save time
    % rho = options.badmm_rho*median(median(pdist2(c.supp', supp', 'sqeuclidean')));
else
    rho = 2.*mean(mean(pdist2(c.supp', supp', 'sqeuclidean')));
end

if isfield(options, 'badmm_tau')
    tau = options.tau;
else
    tau = 10;
end

if isfield(options, 'badmm_tol')
    badmm_tol = options.badmm_tol;
else
    badmm_tol = 1E-4;
end

tstart = clock;
display = 1;          % option of displaying
displayfreq = 1;      % gap of display
checkfreq = 200;      % check frequency
maxtime = inf;        % maximum running time
savewhist = 0;        % save w history
savetimehist = 0;     % save time history

if isfield(options, 'display'),       display = options.display;            end    
if isfield(options, 'displayfreq'),   displayfreq = options.displayfreq;    end    
if isfield(options, 'checkfreq'),     checkfreq = options.checkfreq;        end 
if isfield(options, 'maxtime'),       maxtime = options.maxtime;            end   
if isfield(options, 'savewhist'),     savewhist = options.savewhist;        end    
if isfield(options, 'savetimehist'),  savetimehist = options.savetimehist;  end    
if display == 1
    fprintf('\n-------------- Bregman ADMM ---------------\n');
    fprintf('iter |   feas  |  succhg |  time\n');
end
indrow = 1:sum(stride); indcol = repelem(1:n, 1, stride);
Ma = sparse(indrow, indcol, ones(sum(stride),1));
Fnorm = @(x) norm(x, 'fro');
normw = norm(w);
normcw = norm(c.w); 
normX = Fnorm(X);
normZ = Fnorm(Z); 
normY = Fnorm(Y);
breakyes = 0;
timeout = 0;
whist = []; 
timehist = [];

%% main loop
for iter = 1 : nIter
    
    % record X, Y and Z. 
    Xold = X; 
    Zold = Z; 
    cwold = c.w; 
    Yold = Y;
    normcwold = normcw; 
    normXold = normX; 
    normZold = normZ; 
    normYold = normY;
    
    % update X
    X = Z .* exp((C+Y)/(-rho)) + eps;
    X = bsxfun(@times, X', w'./sum(X)')';
    
    % update Z
    Z = X .* exp(Y/rho) + eps;
    spZ = sparse(spIDX_rows, spIDX_cols, Z(:), support_size * n, m);
    tmp = full(sum(spZ, 2)); 
    tmp = reshape(tmp, [support_size, n]);
    dg = bsxfun(@times, 1./tmp, c.w');
    dg = sparse(1:support_size*n, 1:support_size*n, dg(:));
    Z = full(spIDX * dg * spZ);
    
    % update Y
    Y = Y + rho*(X - Z);
    
    % update c.w
    tmp = bsxfun(@times, tmp, 1./sum(tmp));
    sumW = sum(sqrt(tmp),2)'.^2;
    c.w = sumW/sum(sumW);
    
    if iter == 1 || mod(iter, checkfreq) == 0
        normcw  = norm(c.w); 
        normX   = Fnorm(X);
        normZ   = Fnorm(Z); 
        normY   = Fnorm(Y);
        feas1   = norm(sum(X,1)-w)/(1+normX+normw);
        feas2   = Fnorm(repmat(c.w',[1,n])-Z*Ma)/(1+normcw+normZ);
        feas3   = Fnorm(X-Z)/(1+normX+normZ);
        feas    = max([feas1; feas2; feas3]);
        succhg1 = norm(c.w-cwold)/(1+normcw+normcwold);
        succhg2 = Fnorm(X-Xold)/(1+normX+normXold);
        succhg3 = Fnorm(Z-Zold)/(1+normZ+normZold);
        succhgP = max([succhg1; succhg2; succhg3]);
        succhgD = Fnorm(Y-Yold)/(1+normY+normYold);
        succhg  = max([succhgP; succhgD]);
        
        if (max(feas, succhg) < badmm_tol), 
            breakyes = 1; 
        end
    end
    if savewhist == 1, whist = [whist c.w']; end
    if savetimehist == 1, timehist = [timehist; etime(clock,tstart)]; end
    if (display == 1) && ((mod(iter, displayfreq) == 0) && (mod(iter, checkfreq) == 0) || breakyes)
        fprintf('%5.0f|%0.3e|%0.3e|%3.2e\n', iter, feas, succhg, etime(clock, tstart));
    end
    if (etime(clock, tstart) >= maxtime), timeout = 1; end
    if (breakyes) || (timeout), break; end
end