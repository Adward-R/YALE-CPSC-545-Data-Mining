% function script5
%SCRIPT5   Complete the exercise only using the core MATLAB programming language
%and the following functions as needed: load, rng, size, zeros, randsample,
%randperm figure, imagesc, title, xlabel, ylabel, print, trace, fitctree,
%predict,view

load('leaf.mat');
n_class = 30;
n_samples = size(x, 1);

% Leave-out-one CV part
conf_mat = zeros(n_class, n_class);
cv_round = n_samples;
accuracy = 0;
for i = 1:cv_round
    ind_test = i;
    ind_train = 1:n_samples;
    ind_train(i) = [];
    M = fitctree(x(ind_train, :), c(ind_train));
    c_hat = predict(M, x(ind_test, :));
    conf_mat(c(ind_test), c_hat) = conf_mat(c(ind_test), c_hat) + 1;
    if c(ind_test) == c_hat
        accuracy = accuracy + 1;
    end
end

accuracy = accuracy / n_samples;
imagesc(conf_mat);
title(strcat('Leave-out-one CV: accuracy: ', num2str(accuracy)));
print('Leave-out-one', '-dpng');

% 2-fold CV part
conf_mat = zeros(n_class, n_class);
cv_round = 2;
fold_size = n_samples / cv_round;
accuracy = 0;
for i = 1:cv_round
    ind = randperm(n_samples);
    ind_train = ind(1:fold_size);
    ind_test = ind(fold_size+1:end);
    M = fitctree(x(ind_train, :), c(ind_train));
    c_hat = predict(M, x(ind_test, :));
    c_test = c(ind_test);
    accuracy = accuracy + sum(c_hat == c_test);
    for j = 1:size(c_hat, 1)
        conf_mat(c_test(j), c_hat(j)) = conf_mat(c_test(j), c_hat(j)) + 1;
    end
end

accuracy = accuracy / n_samples;
imagesc(conf_mat);
title(strcat('2-fold CV: accuracy: ', num2str(accuracy)));
print('2-fold', '-dpng');

% 17 fold CV part
conf_mat = zeros(n_class, n_class);
cv_round = 17;
fold_size = n_samples / cv_round;
accuracy = 0;
for i = 1:cv_round
    ind = randperm(n_samples);
    ind_test = ind(1:fold_size);
    ind_train = ind(fold_size+1:end);
    M = fitctree(x(ind_train, :), c(ind_train));
    c_hat = predict(M, x(ind_test, :));
    c_test = c(ind_test);
    accuracy = accuracy + sum(c_hat == c_test);
    for j = 1:size(c_hat, 1)
        conf_mat(c_test(j), c_hat(j)) = conf_mat(c_test(j), c_hat(j)) + 1;
    end
end

accuracy = accuracy / n_samples;
imagesc(conf_mat);
title(strcat('17-fold CV: accuracy: ', num2str(accuracy)));
print('17-fold', '-dpng');
% return
% end
