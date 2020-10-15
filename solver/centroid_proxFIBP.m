%%************************************************************************
%% Call Fast Proximal IBP to compute Wasserstein Barycenter
function [Pi, iter] = centroid_proxFIBP(stride, supp, w, c0, options) 
    
ibp_vareps = options.ibp_vareps; 
ibp_tol = options.ibp_tol; 
iter = 0; 

% initialization
[~, ~, j, u, v] = centroid_FIBP(stride, supp, w, c0, options);
iter = iter + j; 

for i=1:4
    options.ibp_vareps = ibp_vareps/3^i;
    options.ibp_tol = ibp_tol/10^i; 
    options.u = u; 
    options.v = v; 
    [~, Pi, j, u, v] = centroid_FIBP(stride, supp, w, c0, options);
    iter = iter + j; 
end  

end