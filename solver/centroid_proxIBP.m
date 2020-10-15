%%************************************************************************
%% Call Proximal IBP to compute Wasserstein Barycenter
function [Pi, iter] = centroid_proxIBP(stride, supp, w, c0, options) 
    
ibp_vareps = options.ibp_vareps; 
ibp_tol = options.ibp_tol; 
iter = 0; 

% initialization
[~, ~, j, u, v] = centroid_IBP(stride, supp, w, c0, options);
iter = iter + j; 

for i=1:4
    options.ibp_vareps = ibp_vareps/2^i;
    options.ibp_tol = ibp_tol/5^i; 
    options.u = u; 
    options.v = v; 
    [~, Pi, j, u, v] = centroid_IBP(stride, supp, w, c0, options);
    iter = iter + j; 
end  

end