% Problem 4: Complete the exercise only using the core MATLAB language 
% (arithmetic operations, loops, etc.) and the following functions as needed.
%
%   size    size of an array
%   eye     creates an identity matrix
%   ones    creates a vector/matrix of ones
%   svd     computes the Singular Value Decomposition
%   diag    extracts the diagonal from a diagonal matrix
%   figure  creates a figure
%   plot    plots lines or points
%   text    add text description to data points
%   title   add title to plot
%   hold on hold current graphic
%
clear all; close all; clc


% Load data
load('nycity.mat');  % D, c
n = length(c);
J = eye(n) - (1/n) * ones(n, 1) * ones(1, n);
B = -0.5 * J * D * J;
[U, S, V] = svd(B);
X = U(:, 1:2) * (S(1:2, 1:2).^0.5);

% [X, E] = cmdscale(D);
plot(X(:, 1), X(:, 2), '.', 'MarkerSize', 30);
% plot(X(1, :), X(2, :), '.', 'MarkerSize', 30);

text(X(:, 1)+.1, X(:, 2), c, 'Color', 'r');


