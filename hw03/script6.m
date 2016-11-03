function script6
%SCRIPT6   Complete the exercise only using the code MATLAB language and the
%following functions as neeeded: load, size, zeros, logical, trace, repmat,
%fprintf, figure, hold all, plot, legend

    % c [class labels] mx1 vec of int>0
    % nc [num of classes] int;
    % x [attribute matrix, data]
    load('seed_data.mat');
    nk = 6;
    rng(0);
    Md = naivebayes_train(c, nc, x, nk);
    c_hat = naivebayes_classify(Md, x(:, 1:end-1));

    % Leave-out-one CV part
    n_samples = size(x, 1);
    conf_mat = zeros(nc, nc);
    cv_round = n_samples;
%    rand_ind = randperm(n_samples);
%    x = x(rand_ind, :);
%    c = c(rand_ind);
    for i = 1:cv_round
        ind_test = i;
        ind_train = 1:n_samples;
        ind_train(i) = [];
        Md = naivebayes_train(c(ind_train), nc, x(ind_train, :), nk);
        c_hat = naivebayes_classify(Md, x(ind_test, :));
        conf_mat(c(ind_test), c_hat) = conf_mat(c(ind_test), c_hat) + 1;
    end

    accuracy = sum(trace(conf_mat)) / n_samples
    conf_mat

    % Distribution Visualization
    Md = naivebayes_train(c, nc, x, nk);
    bin_width = (Md.attr_bounds(:, 2, :) - Md.attr_bounds(:, 1, :)) / Md.nk;
    for j = [1, 2]
        close all;
        figure;
        hold all;
        for ic = 1:nc
            x_coors = (0.5:1:5.5) * bin_width(j, 1, ic) + Md.attr_bounds(j, 1, ic);
            y_coors = Md.pr(j, :, ic) * Md.pc(ic);
            plot(x_coors, y_coors);
        end
        clabels = {};
        for ic = 1:nc
            clabels(ic) = {strcat('class-', int2str(Md.labels(ic)))};
        end
        legend(clabels);
        print(strcat('curve-', int2str(j)), '-dpng')
    end

return
end


function Md = naivebayes_train(c,nc,x,nk)
%NAIVEBAYES_TRAIN   Train a Naive Bayes Classifier using histogram with k-bins.

    [m, n] = size(x);
    epsilon = 10^(-12);

    labels = unique(c);
    f_c = zeros(nc, 1);
    p_c = zeros(nc, 1);
    for k = 1:nc
        f_c(k) = sum(c == labels(k));
        p_c(k) = f_c(k) / m;
    end

    attr_bounds = zeros(n, 2, nc); % min & max for each variable IN EACH CLASS
    p_xj_c = zeros(n, nk, nc);
    for j = 1:n
        for ic = 1:nc
            attr_bounds(j, 1, ic) = min(x(c == labels(ic), j));
            attr_bounds(j, 2, ic) = max(x(c == labels(ic), j));
            bin_width = (attr_bounds(j, 2, ic) - attr_bounds(j, 1, ic)) / nk;
            for k = 1:nk
                low_lim = attr_bounds(j, 1, ic) + bin_width * (k - 1);
                hih_lim = attr_bounds(j, 1, ic) + bin_width * k;
                combined_freq = sum( (x(:, j) - low_lim >= epsilon) & (hih_lim - x(:, j) > epsilon) & (c == labels(ic)) );
                p_xj_c(j, k, ic) = (combined_freq + 1) / (f_c(ic) + 1); % Laplace Correction
            end
        end
    end

    Md.pr = p_xj_c;
    Md.pc = p_c;
    Md.attr_bounds = attr_bounds;
    Md.nk = nk;
    Md.labels = labels;
end

function b = naivebayes_classify(Md,y)
%NAIVEBAYES_TEST   Classify attribute data Y using Naive Bayes model PR.

    [k, n] = size(y);
    b = zeros(k, 1);
    nc = length(Md.labels);
    low_bounds = min(Md.attr_bounds(:, 1, :), [], 3);
    hih_bounds = max(Md.attr_bounds(:, 2, :), [], 3);
    bin_width = (hih_bounds - low_bounds) / Md.nk;

    for iy = 1:k
        % discretize the test data
        row = zeros(n, 1);
        for j = 1:n
            row(j) = ceil((y(iy, j) - low_bounds(j)) / bin_width(j));
            if row(j) < 1
                row(j) = 1;
            elseif row(j) > Md.nk
                row(j) = Md.nk;
            end
        end

        pr_hat = Md.pc;
        for ic = 1:nc
            for j = 1:n
                pr_hat(ic) = pr_hat(ic) * Md.pr(j, row(j), ic);
            end
        end
        b(iy) = Md.labels(find(pr_hat == max(pr_hat)));
    end

end

