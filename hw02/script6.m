% Problem 6: Complete the exercise using the core MATLAB language (arithmetic,
% loops, etc.) and the following functions
%
%   size        size of an array
%   mean        computes the average or mean value
%   svd         computes the singular value decomposition
%   figure      creates a figure
%   imshow      plots an image
%   reshape     reshape array
%   title       add title to plot
%   scatter3    create 3-dimensional scatter plot
%   pdist2      pairwise distance
%   eye         identity matrix
%   ones        vector of ones
%
clear all; close all; clc;


% Load data
load('bunny.mat');


% Example: plotting bunny
% figure;
% imshow(reshape(X(1,:),sz),'initialmagnification','fit');
% title('First bunny');
% print('first_bunny.png','-dpng');


% Example: making 3d scatter plot
% figure;
% scatter3(1:10,1:10,1:10,24,1:10);
% colormap('jet')
% print('3d_scatter.png','-dpng');

m = size(X, 1);

% 2
mean_img = mean(X);

% 3
figure;
imshow(reshape(mean_img, sz), 'initialmagnification', 'fit');

% 4
title('Mean bunny');
print('mean_bunny.png','-dpng');

% 5
J = eye(m) - (1/m) * ones(m, 1) * ones(1, m);
Xmean = J * X;

% 6
[U, S, V] = svd(Xmean, 'econ');  % cols of V are eigenvectors
eigenVs = diag(S.^2);

% 7 
Vreduce = V(:, 1:3); 
Xproj = X * Vreduce;

% 8
figure;
scatter3(Xproj(:, 1), Xproj(:, 2), Xproj(:, 3), 24, theta);

% 9
title('PCA Coordinates');
print('pca_coordinates.png','-dpng');

% 10
D = pdist2(X, X);
B = -1/2 * J * D.^2 * J;
[U, S, V] = svd(B);
% Force the first coordinate of each vector to be positive.
for i = 1:m
    U(i, 3) = - U(i, 3);
end
Y = U * S.^0.5;

% 11
figure;
scatter3(Y(:, 1), Y(:, 2), Y(:, 3), 24, theta);

% 12
title('MDS Coordinates');
print('mds_coordinates.png','-dpng');


