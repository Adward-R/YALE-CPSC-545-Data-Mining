function script6

    % Test DBScan on unlabeled data
    data = load('clusters1.mat');
    x = data.x;
    x = x(randperm(size(x, 1)), :);
    D = pdist2(x, x);
    N = size(x, 1);

    epsilon = pctl(reshape(D, [N^2, 1]), 1);
    min_pts = pctl(sum(D <= epsilon, 2), 10);
    [C, point_type] = dbscan(x, min_pts, epsilon, @pdist2);
    scatter(x(:, 1), x(:, 2), 30, C);
    title('Colored by Clusters, Eps=1.0%');
    print('6-1', '-dpng');
    scatter(x(:, 1), x(:, 2), 30, point_type);
    title('Colored by Point-types, Eps=1.0%');
    print('6-2', '-dpng');

    epsilon = pctl(reshape(D, [N^2, 1]), 0.3);
    min_pts = pctl(sum(D <= epsilon, 2), 10);
    [C, point_type] = dbscan(x, min_pts, epsilon, @pdist2);
    scatter(x(:, 1), x(:, 2), 30, C);
    title('Colored by Clusters, Eps=0.3%');
    print('6-3', '-dpng');
    scatter(x(:, 1), x(:, 2), 30, point_type);
    title('Colored by Point-types, Eps=0.3%');
    print('6-4', '-dpng');

    close all;

    % Test DBScan on labeled real data
    data = load('clusters2.mat');
    x = data.x;
    c = data.c;
    N = size(x, 1);
    n_rnd = 27;
    [best_epsilon, worst_epsilon, best_min_pts, worst_min_pts, best_ri] = deal(0);
    worst_ri = 1;
    [best_x, worst_x, best_C, worst_C] = deal([]);
    for pctl_eps = [3, 5, 7],
        for pctl_pts = [10, 20, 30],
            for i_perm = 1:3,
                xr = x(randperm(N), :);
                D = pdist2(xr, xr);
                epsilon = pctl(reshape(D, [N^2, 1]), pctl_eps);
                min_pts = pctl(sum(D <= epsilon, 2), pctl_pts);
                [C, ~] = dbscan(xr, min_pts, epsilon, @pdist2);
                ri = RandIndex(c, C);
                if ri > best_ri,
                    best_ri = ri;
                    best_epsilon = epsilon;
                    best_min_pts = min_pts;
                    best_x = xr;
                    best_C = C;
                end
                if ri < worst_ri,
                    worst_ri = ri;
                    worst_epsilon = epsilon;
                    worst_min_pts = min_pts;
                    worst_x = xr;
                    worst_C = C;
                end
            end
        end
    end

    best_ri
    best_epsilon
    best_min_pts
    worst_ri
    worst_epsilon
    worst_min_pts
    
    scatter(best_x(:, 1), best_x(:, 2), 30, best_C);
    title('Best RandIndex Config Clusters')
    print('6-5', '-dpng');
    scatter(worst_x(:, 1), worst_x(:, 2), 30, worst_C);
    title('Worst RandIndex Config Clusters')
    print('6-6', '-dpng');
    close all;


    function [epsilon] = pctl(v, percent)
        sv = sort(v);
        m = size(v, 1);
        % same as `quantile(v, percent/100)`
        epsilon = interp1q([0 (0.5:(m-0.5))./m 1]', sv([1 1:m m], :), percent / 100');
        return
    end

end

function [C, point_type] = dbscan(X, min_pts, epsilon, dist)
    C_ind = 0;
    N = size(X, 1);
    C = zeros(N, 1);
    D = dist(X, X);
    
    visited = false(N, 1);
    % -1 for noise point, 0 for border point, and 1 for core point
    point_type = zeros(N, 1);
    
    for i = 1:N,
        if ~visited(i),
            nbrs = find(D(i,:) <= epsilon);
            if numel(nbrs) < min_pts,
                point_type(i) = -1; % X(i,:) is NOISE
            else,
                point_type(i) = 1;
                visited(i) = true;
                C_ind = C_ind + 1;  % start a new clusters
                ExpandCluster(i, nbrs, C_ind);
            end
        end
    end
    
    % Add all unvisited density-reachable points from X(i, :) -> C_ind
    function ExpandCluster(i, nbrs, C_ind)
        C(i) = C_ind;
        
        k = 1;  % queue head index
        while true,
            j = nbrs(k);
            
            if ~visited(j),
                C(j) = C_ind;
                visited(j) = true;
                new_nbrs = find(D(j, :) <= epsilon);
                if numel(new_nbrs) < min_pts,
                    point_type(j) = 0;  % mark y as border point
                else,
                    point_type(j) = 1;
                    nbrs = [nbrs new_nbrs];
                end
            end
            % if C(j) == 0,
            %    C(j) = C_ind;
            % end
            
            k = k + 1;
            if k > numel(nbrs),  % queue is empty
                break;
            end
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
