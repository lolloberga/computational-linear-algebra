clear all
close all
clc

% Utils functions
threshold_for_considering_eig_zero = @(eigD) mean(eigD)/2.5;

% Load dataset
spiral_dataset = load('dataset/Spiral.mat');
circle_dataset = load('dataset/Circle.mat');

% Dataset analysis
spiral = spiral_dataset.X(:, 1:2);
circle = circle_dataset.X(:, :);

figure
tiledlayout(2,1)
% Sprial plot
nexttile
scatter(spiral(:, 1), spiral(:, 2))
title('Spiral')

% Circle plot
nexttile
scatter(circle(:, 1), circle(:, 2))
title('Circle')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sigma = 1.0;
k_values = [10, 20, 40];

figure
tiledlayout(3,1)
for k = k_values
    s=sprintf('K-NN with K = %d', k);
    disp(s);

    % Point 1: construct the k-nearest neighborhood similarity graph and its adjacency matrix W
    W = similarity(spiral, k, sigma);

    % Point 2: compute the degree matrix D and the Laplacian matrix L
    d=sum(W,2);
    D=diag(d);
    L = D - W;
    
    % Point 3: compute the number of connected components of the similarity graph
    [eigV, eigD] = eigs(L, 10, 'smallestabs');
    eigD=diag(eigD);
    [eigD,IJ]=sort(eigD);
    eigV=eigV(:,IJ);
    
    i0 = sum(eigD < 1.0e-6);
    s=sprintf('The graph has %d connected components', i0);
    disp(s);
    
    % Point 4: Compute some small eigenvalues of L and use their values to choose a suitable number of clusters M for the points data-sets
    thrsh = threshold_for_considering_eig_zero(eigD);
    m_clusters = sum(eigD < thrsh);
    
    % Point 5: ompute the M eigenvectors that correspond to the M smallest eigenvalues of the Laplacian matrix
    [U, eigD] = eigs(L, m_clusters, 'smallestabs');
    
    % Point 6: let yi ∈ RM be the vector corresponding to the i-th row of U. Cluster the points in yi with the k-means algorithm into clusters C1, ..., CM
    idx = kmeans(U, m_clusters);
    
    % Point 7: assign the original points in X to the same clusters as their corresponding rows in U
    sprial_copy = repmat(spiral_dataset.X, 1);
    sprial_copy(:, 3) = idx';
    
    % Point 8: plot the clusters of points X with different colors
    nexttile
    gscatter(sprial_copy(:, 1), sprial_copy(:, 2), sprial_copy(: ,3));
    title(sprintf('Cluster with initial K-NN with K=%d', k));
end

% Point 9: Compute and plot clusters for the same set of points using k-means directly on the initial points

% Spiral dataset
spiral_copy = repmat(spiral_dataset.X, 1);
best_spiral_idx = find_best_clusters(spiral_copy(:, 1:2), 10);
spiral_copy(:, 3) = best_spiral_idx';
% Circle
circle_copy = repmat(circle, 1);
best_circle_idx = find_best_clusters(circle_copy, 10);
circle_copy = [circle_copy best_circle_idx];

figure
tiledlayout(2,1)

nexttile
gscatter(spiral_copy(:, 1), spiral_copy(:, 2), spiral_copy(: ,3));
title('Spiral: cluster with optimal K clusters');

nexttile
gscatter(circle_copy(:, 1), circle_copy(:, 2), circle_copy(: ,3));
title('Circle: cluster with optimal K clusters');
