%%************************************************************************
%% Call IBP to compute Wasserstein Barycenter
function [c, T, iter, u, v, whist, timehist] = centroid_IBP(stride, supp, w, c0, options) 
% The algorithmic prototype of Wasserstein Barycenter using IBP

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
spIDX_rows   = zeros(support_size * m, 1);
spIDX_cols   = zeros(support_size * m, 1);

for i = 1:n
    [xx, yy]        = meshgrid((i-1)*support_size + (1:support_size), posvec(i):posvec(i+1)-1);
    ii              = support_size*(posvec(i)-1) + (1:(support_size*stride(i)));
    spIDX_rows(ii)  = xx';
    spIDX_cols(ii)  = yy';
end

spIDX = repmat(speye(support_size), [1, n]);

% initialization
if isfield(options, 'u')
    u = options.u; 
else
    u = ones(n*support_size, 1); 
end
if isfield(options, 'v')
    v = options.v; 
else
    v = ones(m, 1); 
end

if isfield(options, 'dist_matrix')
    C = options.dist_matrix;
else
    C = pdist2(c.supp', supp', 'sqeuclidean');
end

nIter = 20000;
if isfield(options, 'ibp_max_iters')
    nIter = options.ibp_max_iters;
end
  
if isfield(options, 'ibp_vareps')
    rho = options.ibp_vareps;   % compute rho in advance to save time
    % rho = options.ibp_vareps * median(median(pdist2(c.supp', supp', 'sqeuclidean')));
else
    rho = 0.01 * median(median(pdist2(c.supp', supp', 'sqeuclidean')));
end
  
if isfield(options, 'ibp_tol')
    ibp_tol = options.ibp_tol;
else
    ibp_tol = 1E-4; % no updates of support
end
  
xi            = exp(-C/rho);
xi(xi<1e-200) = 1e-200; 
xi            = sparse(spIDX_rows, spIDX_cols, xi(:), support_size*n, m);
X             = spdiags(u, 0, n*support_size, n*support_size)*xi*spdiags(v, 0, m, m); 

tstart        = clock;
display       = 1;            % option of displaying
displayfreq   = 1;            % gap of display
checkfreq     = 200;          % check frequency
maxtime       = inf;          % maximum running time
savewhist     = 0;            % save w history
savetimehist  = 0;            % save time history
if isfield(options, 'display'),       display = options.display;            end    
if isfield(options, 'displayfreq'),   displayfreq = options.displayfreq;    end    
if isfield(options, 'checkfreq'),     checkfreq = options.checkfreq;        end 
if isfield(options, 'maxtime'),       maxtime = options.maxtime;            end   
if isfield(options, 'savewhist'),     savewhist = options.savewhist;        end    
if isfield(options, 'savetimehist'),  savetimehist = options.savetimehist;  end    
if display == 1
    fprintf('\n-------------- IBP ---------------\n');
    fprintf('iter |   feas  |  succhg |  time\n');
end
Fnorm = @(x) norm(x, 'fro');
normw = norm(w);
normcw = norm(c.w); 
normX = Fnorm(X); 
normu = norm(log(u));
normv = norm(log(v)); 
breakyes = 0;
timeout = 0;
whist = []; 
timehist = [];

%% main loop
for iter = 1:nIter 
    
    % record. 
    Xold = X;  
    cwold = c.w; 
    uold = u;
    vold = v;
    normcwold = normcw; 
    normXold = normX; 
    normuold = normu; 
    normvold = normv;
    
    % main update rule. 
    w0  = repmat(c.w', n, 1);
    u   = w0./full(xi*v);
    v   = w'./full(xi'*u);
    c.w = geomean(reshape(u.*full(xi*v), support_size, n), 2)';
    X   = spdiags(u, 0, n*support_size, n*support_size)*xi*spdiags(v, 0, m, m); 
    
    if iter == 1 || mod(iter, checkfreq) == 0
        normcw = norm(c.w); 
        normX = Fnorm(X);
        normu = norm(log(u)); 
        normv = norm(log(v));
        feas1 = norm(sum(X, 1)-w)/(1+normX+normw);
        feas2 = norm(sum(X, 2)-w0)/(1+normcw+normX);
        feas = max([feas1; feas2]);
        succhg1 = norm(c.w-cwold)/(1+normcw+normcwold);
        succhg2 = Fnorm(X-Xold)/(1+normX+normXold);
        succhgP = max([succhg1; succhg2]);
        succhg3 = norm(log(u)-log(uold))/(1+normu+normuold); 
        succhg4 = norm(log(v)-log(vold))/(1+normv+normvold); 
        succhgD = max([succhg3; succhg4]);
        succhg = max([succhgP; succhgD]);
        T = full(spIDX*X); 
        if (max(feas, succhg) < ibp_tol), breakyes = 1; end
    end
    if savewhist == 1, whist = [whist c.w']; end
    if savetimehist == 1, timehist = [timehist; etime(clock,tstart)]; end
    if (display == 1) && ((mod(iter, displayfreq) == 0) && (mod(iter, checkfreq) == 0) || breakyes)
        fprintf('%5.0f|%0.3e|%0.3e|%3.2e\n', iter, feas, succhg, etime(clock,tstart));
    end
    if (etime(clock,tstart) >= maxtime), timeout = 1; end
    if (breakyes) || (timeout), break; end
end