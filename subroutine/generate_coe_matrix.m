%%************************************************************************
%% This code is to generate the coefficients A, b, lb (lower bd) for LP 
%% arising in computing Wasserstein barycenter   
%%************************************************************************
function [Aeq, beq, lb] = generate_coe_matrix(N, m, mt, wt) 

% wt = [w1;w2;...;wN] should be a vertial vector
if size(wt, 2) ~= 1, wt = wt'; end

%% generate A
% count row id from left to right, top to bottom
% row_id = []; col_id = []; len_id = [];
% for k = 1 : N
%     for i = 1 : m
%         tmp = m + sum(mt(1:k-1))*m + (i-1) + [1:m:(mt(k)*m)];
%         col_id = [col_id i tmp];
%         row_id = [row_id (i+(k-1)*m)*ones(1,mt(k)+1)];
%         len_id = [len_id length(col_id)];
%     end
% end
% for k = 1 : N
%     for i = 1 : mt(k)
%         tmp = m + sum(mt(1:k-1))*m + (i-1)*m + [1:m];
%         col_id = [col_id tmp];
%         row_id = [row_id (N*m+sum(mt(1:k-1))+i)*ones(1,m)];
%     end
% end
% row_id = [row_id (N*m+sum(mt)+1)*ones(1,m)];
% col_id = [col_id 1:m];
% v = ones(1, length(row_id)); v(1) = -1;
% for i = 1 : length(len_id)-1, v(len_id(i)+1) = -1; end
% Aeq = sparse(row_id, col_id, v);

% generate by Kronecker product
TMP = [];
for k = 1 : N
    TMP = sparse(blkdiag(TMP, ones(1,mt(k))));
end
A1 = [repmat(-speye(m),N,1) kron(TMP, speye(m))];
A2 = kron(circshift(speye(sum(mt)+1),1,2), ones(1,m));
Aeq = [A1; A2];

%% generate b and lb
beq = [zeros(N*m,1); wt; 1];
lb = zeros(m+m*sum(mt), 1);



