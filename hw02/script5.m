% Problem 5: Complete the exercise only using the core features of the MATLAB
% language (arithmetic operations, loops, array, etc.) and any of the folowing
% functions
%
%   rng         Seed random number generator
%   cos         cosine
%   sin         sine
%   rand        uniformly random points on [0,1]
%   randn       random normal mean 0 variance 1 points
%   mean        computes average or mean value
%   ones        creates a vector of ones
%   repmat      replicate and tile an array
%   inv         computes the inverse of array
%   sqrt        square root
%   svd         computes the singular value decomposition
%   figure      creates a figure
%   scatter     creates scatter plot
%   title       creates title
%   colormap    define colormap    
%   hold on     hold current graph
%   axis equal  equal aspect ratio
%   quiver      quiver plot
%   
clear all; close all; clc; 
% Seed the random number generator to you favorite integer, e.g., rng(213);
rng(2);

% Part A: Data Generation
m = 1000;
n = 2;
x = 10.*rand(m,1);
theta = pi / 6; % rand() * pi;
X = [x.*cos(theta) x.*sin(theta)];
% X = imnoise(X, 'gaussian', 0.2, 0.2);
X = X + randn(1000, 2);
% plot(X(:,1), X(:,2), '.', 'MarkerSize', 30);

% Part B: Mahalanobis and Principal Components
means = mean(X);
Sigma = (n-1)^(-1) * X' * X;
dist2mean = zeros(m, 1);
for i = 1:m,
    tmp = X(i, :) - means;
    dist2mean(i) = (tmp * inv(Sigma) * tmp').^0.5;
end
% D = (X * Cov * X').^0.5;

X_hat = X - (1/m) * ones(m, 1) * ones(1, m) * X;  % mean extraction
Cov = (m-1)^(-1) * (X_hat' * X_hat);
[U, S, V] = svd(Cov);
% obtain eigenvectors
fai1 = U(:, 1);
fai2 = U(:, 2);
% obtain eigenvalues
lambda1 = S(1, 1);
lambda2 = S(2, 2);

% Part C: Visualization
fai1 = lambda1.^0.5 * fai1;
fai2 = lambda2.^0.5 * fai2;
scatter(X(:, 1), X(:, 2), 9, dist2mean, 'filled');
hold on;
quiver(means(1), means(2), fai1(1), fai1(2), 'k', 'LineWidth', 4);
quiver(means(1), means(2), fai2(1), fai2(2), 'k', 'LineWidth', 4);
print('mahal_pca_illustration', '-dpng');