%%************************************************************************
%% Call Fast IBP to compute Wasserstein Barycenter
function [c, T, iter, whist, timehist] = centroid_FIBP(stride, supp, w, c0, options) 
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
u_tilde = zeros(n*support_size, 1); 
u       = u_tilde; 
v_tilde = zeros(m, 1);
v       = v_tilde; 
theta   = 1;

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
    ibp_tol = 1E-6; % no updates of support
end
  
xi              = exp(-C/rho);
xi(xi<1e-200)   = 1e-200; % add trick to avoid program breaking down
xi              = sparse(spIDX_rows, spIDX_cols, xi(:), support_size * n, m);
X               = spdiags(exp(u), 0, n*support_size, n*support_size)*xi*spdiags(exp(v), 0, m, m);

tstart          = clock;
display         = 1;            % option of displaying
displayfreq     = 1;            % gap of display
checkfreq       = 200;          % check frequency
maxtime         = inf;          % maximum running time
savewhist       = 0;            % save w history
savetimehist    = 0;            % save time history
if isfield(options, 'display'),       display = options.display;            end    
if isfield(options, 'displayfreq'),   displayfreq = options.displayfreq;    end    
if isfield(options, 'checkfreq'),     checkfreq = options.checkfreq;        end 
if isfield(options, 'maxtime'),       maxtime = options.maxtime;            end   
if isfield(options, 'savewhist'),     savewhist = options.savewhist;        end    
if isfield(options, 'savetimehist'),  savetimehist = options.savetimehist;  end    
if display == 1
    fprintf('\n-------------- Fast IBP ---------------\n');
    fprintf('iter |   feas  |  succhg |  time\n');
end
Fnorm = @(x) norm(x, 'fro');
normw = norm(w);
normcw = norm(c.w); 
normX = Fnorm(X); 
normu = norm(u);
normv = norm(v); 
breakyes = 0;
timeout = 0;
whist = []; 
timehist = [];

%% main loop
for iter = 1:nIter   
    % record all the variables. 
    Xold      = X;  
    cwold     = c.w; 
    uold      = u;
    vold      = v;
    normcwold = normcw; 
    normXold  = normX; 
    normuold  = normu; 
    normvold  = normv;
    
    % main update rule. 
    u_bar       = (1-theta)*u + theta*u_tilde; 
    v_bar       = (1-theta)*v + theta*v_tilde;
    
    X_bar       = spdiags(exp(u_bar), 0, n*support_size, n*support_size)*xi*spdiags(exp(v_bar), 0, m, m); 
    r_X_bar     = sum(X_bar, 2)/sum(X_bar(:));
    c_X_bar     = sum(X_bar, 1)/sum(X_bar(:));
    
    tmp         = u_tilde - r_X_bar/(4*theta); 
    u_tilde_new = tmp - mean(tmp);
    u_hat       = u_bar + theta*(u_tilde_new - u_tilde);
    u_tilde     = u_tilde_new; 
    v_tilde     = v_tilde - c_X_bar'/(4*theta); 
    v_hat       = v_bar - c_X_bar'/4; 
    
    % heuristic: remove some steps in the pseudocode for saving the computational cost
    if w*v > w*v_hat - log(exp(u_hat)'*xi*exp(v_hat))
        u_aux   = u; 
        v_aux   = v;        
    else
        u_aux   = u_hat; 
        v_aux   = v_hat;
    end
    
    tmpw  = geomean(reshape(exp(u_aux).*full(xi*exp(v_aux)), support_size, n), 2)';
    w0    = repmat(tmpw', n, 1);
    u     = log(w0./full(xi*exp(v_aux)));
    v     = log(w'./full(xi'*exp(u)));  
    c.w   = geomean(reshape(exp(u).*full(xi*exp(v)), support_size, n), 2)';
    X     = spdiags(exp(u), 0, n*support_size, n*support_size)*xi*spdiags(exp(v), 0, m, m);    
    theta = theta*(sqrt(theta*theta+4)-theta)/2;  
    
    if iter == 1 || mod(iter, checkfreq) == 0
        normcw = norm(c.w); 
        normX = Fnorm(X);
        normu = norm(u); 
        normv = norm(v);
        feas1 = norm(sum(X, 1)-w)/(1+normX+normw); 
        feas2 = norm(sum(X, 2)-repmat(c.w', n, 1))/(1+normcw+normX); 
        feas = max([feas1; feas2]);
        succhg1 = norm(c.w-cwold)/(1+normcw+normcwold);
        succhg2 = Fnorm(X-Xold)/(1+normX+normXold);
        succhgP = max([succhg1; succhg2]);
        succhg3 = norm(u-uold)/(1+normu+normuold); 
        succhg4 = norm(v-vold)/(1+normv+normvold); 
        succhgD = max([succhg3; succhg4]);
        succhg = max([succhgP; succhgD]);
        T = full(spIDX * X); 
        if (max(feas, succhg) < ibp_tol), breakyes = 1; end
    end
    if savewhist == 1, whist = [whist c.w']; end
    if savetimehist == 1, timehist = [timehist; etime(clock, tstart)]; end
    if (display == 1) && ((mod(iter, displayfreq) == 0) && (mod(iter, checkfreq) == 0) || breakyes)
        fprintf('%5.0f|%0.3e|%0.3e|%3.2e\n', iter, feas, succhg, etime(clock, tstart));
    end
    if (etime(clock, tstart) >= maxtime), timeout = 1; end
    if (breakyes) || (timeout), break; end
end