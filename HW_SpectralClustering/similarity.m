function [W] = similarity(X, K, sigma)

%SIMILARITY Calculates pairwise distances between a set of points using the
%given formula and build the adj matrix.
%   X: matrix containing points' coordinates; rows represent points, and columns represent coordinates.
%   sigma (float): Value of sigma.
%   k (int): Number of nearest neighbors to consider.

% norm2 = @(X) (X(1,:).*X(1,:) + X(2,:).*X(2,:));
% sim_func = @(X1,X2,sig) exp(-norm2(X2-X1)./2*sig^2);
% 
% [N,~]=size(X);
% W=spalloc(N,N,N*K*2);
% 
% for i=1:N
%     sij = sim_func(X, X(i,:), sigma);
%     [vals, idx] = maxk(sij, K+1);
%     W(i, idx(idx ~= i)) = vals(idx ~= i);
%     W(idx(idx ~= i), i) = vals(idx ~= i);
% end

W = sparse([]);
[N,~]=size(X);

for i=1:N
    [Idx1, D1] = knnsearch(X, X(i,:), 'K', K, 'Distance', 'euclidean');
    %delete the reference with the point X(i,:) itself
    Idx1(1)=[];
    D1(1)=[];

    D1 = exp(-(D1.^2)/2*sigma^2);
    W(i,Idx1) = D1;
end

for i=1:N
    for j=1:N
        if W(i,j)~=0 %if in the position (i,j) of W the weight is !=0, so 
                     %also the point j has the point i as a neighbour. 
            W(j,i)=W(i,j); 
        end
    end  
end

end

