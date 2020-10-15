function feas = compfeas(Pi, w, wt, N, mt) 
% compute the feasiblity of primal prolem

if size(wt, 2) ~= 1, wt = wt'; end
if size(mt, 2) ~= 1, mt = mt'; end

Fnorm = @(x) norm(x, 'fro');
indrow = 1:sum(mt); indcol = repelem(1:N, 1, mt);
Ma = sparse(indrow, indcol, ones(sum(mt),1));

normw = norm(w); 
normwt = norm(wt);
normPi = Fnorm(Pi); 

feas1 = norm(sum(Pi, 1)' - wt)/(1 + normPi + normwt);
feas2 = Fnorm(repmat(w,[1, N]) - Pi*Ma)/(1 + normw + normPi);
feas3 = (abs(sum(w)-1) + norm(min(w,0)))/(1 + normw);
feas4 = norm(min(Pi(:),0))/(1+normPi);
feas = max([feas1, feas2, feas3, feas4]);