clear; close all; clc

f = csvread('wine.data');
f = f(1:2, 2:14);

nPoints = 2;
nDimensions = 13;
% P = rand(nDimensions, nPoints);
pointNames = arrayfun( @(i)sprintf('p_{%d}', i),...
1:nPoints, 'UniformOutput', false);
hist
radarPlot(f', 'o:','LineWidth', 1.5, 'MarkerFaceColor', [0,0,0])
legend(pointNames{:}, 'Location', 'Best');
title('Radar Plot Demo');

% figure;
% imagesc(a);