function kmeans_example()

    data = load('iris_num.mat');
    x = data.x;
    c = data.c;

    [N, n] = size(x);
    k = 3;  % num of clusters
    % Illustrate clustering on a two-dimensional dataset
    [C, M, C0, M0] = KmeanClustering(x(:, 1:2), k);
    figure;
    hold all;
    scatter(x(:, 1), x(:, 2), 50, C0);
    scatter(M0(:, 1), M0(:, 2), 50, '+');
    print('5-1', '-dpng');
    hold off;

    figure;
    hold all;
    scatter(x(:, 1), x(:, 2), 50, C);
    scatter(M(:, 1), M(:, 2), 50, '+');
    print('5-2', '-dpng');
    hold off;

    close all;

    % Test clustering quality on the full iris data
    labels = unique(c);
    for i_round = 1:3,
        [C, M, C0, M0] = KmeanClustering(x, k);
        acc = ClusterAccuracy(k, c, C)
    end

    for i_round = 1:3,
        [C, ~, ~, ~] = KmeanClustering(x, k);
        ri = RandIndex(c, C)
    end

    % Compare accuracy, RandIndex, and SSE as cluster quality measures
    n_round = 50;
    RI = zeros(n_round, 1);
    ACC = zeros(n_round, 1);
    SSE = zeros(n_round, 1);
    for i_round = 1:n_round,
        [C, M, ~, ~] = KmeanClustering(x, k);
        RI(i_round) = RandIndex(c, C);
        ACC(i_round) = ClusterAccuracy(k, c, C);
        SSE(i_round) = TotalSSE(C, M, x);
    end

    histogram(ACC, 4);
    title('Accuracy distribution over 50 runs');
    print('5-3', '-dpng');

    histogram(SSE, 4);
    title('SSE distribution over 50 runs');
    print('5-4', '-dpng');

    scatter(ACC, SSE, 30, 'o');
    xlabel('Accuracy');
    ylabel('SSE');
    title('Accuracy v.s. SSE');
    print('5-5', '-dpng');

    close all;
return
end

function [C, M, C0, M0] = KmeanClustering(X, k)
    [N, n] = size(X);
    rand_ind = randperm(N);
    M = X(rand_ind(1:k), :);
    M0 = M;
    C = zeros(N, 1);
    C0 = [];

    for i_round = 1:20,  % maximum rounds
        changed = 0;
        % 1-NN Classification
        for i = 1:N,
            new_dist = ones(k, 1);
            for j = 1:k,
                new_dist(j) = sqrt(sum((X(i, :) - M(j, :)).^2));
            end
            [~, I] = min(new_dist);
            if I ~= C(i),
                changed = changed + 1;
                C(i) = I;
            end
        end
        if i_round == 1,
            C0 = C;
        end

        if changed == 0,
            break;
        end

        % Update centroids
        for j = 1:k,
            M(j, :) = mean(X(C == j, :));  % mean by each column
        end

    end

return
end

function [ri] = RandIndex(L, C)
    N = length(L);
    ri = 0;
    for i = 1:N-1,
        for j = i+1:N,
            if L(i) == L(j) & C(i) == C(j),
                ri = ri + 1;
            elseif L(i) ~= L(j) & C(i) ~= C(j),
                ri = ri + 1;
            end
        end
    end
    ri = (ri * 2) / (N * (N-1));
return
end

% num of expected labels, actual labels, predicted labels
function [acc] = ClusterAccuracy(k, labels, C)
    % label_map = zeros(k, 1);
    % for j = 1:k,
    %     label_map(j) = mode(labels(C == j));
    % end
    label_maps = perms(unique(labels));
    accs = zeros(size(label_maps, 1), 1);
    for i = 1:size(label_maps, 1),
        C_mapped = arrayfun(@(t) label_maps(i, t), C);
        accs(i) = sum(labels == C_mapped) / length(C);
    end
    acc = max(accs);
    return
end

% clusters and centroids, without original labels
function [sse] = TotalSSE(C, M, x)
    sse = 0;
    k = size(M, 1);
    for i = 1:size(x, 1),
        sse = sse + sum((x(i, :) - M(C(i))).^2);
    end
    return
end


