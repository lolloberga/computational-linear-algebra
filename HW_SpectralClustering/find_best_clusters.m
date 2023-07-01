function [classes] = find_best_clusters(X, MAX_K)

%find_best_clusters: Summary of this function goes here
%   Detailed explanation goes here

%the number of clusters you want to try
klist=1:MAX_K;

kmeansfunc = @(X,K)(kmeans(X, K));

eva=evalclusters(X, kmeansfunc, 'CalinskiHarabasz', 'klist', klist);
classes=kmeans(X, eva.OptimalK);

end

