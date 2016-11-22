function script4
    addpath('libsvm-3.21/matlab/', 0)
    GRID_LEN = 30;

   % Linear SVM with slack data
    data = load('mixed.mat');
    c = data.c;
    x = data.x;
    [m, n] = size(x);

    figure;
    hold all;
    scatter(x(c==1, 1), x(c==1, 2), 'r');
    scatter(x(c==-1, 1), x(c==-1, 2), 'b');
    title('MIXED data points colored by their classes');
    print('4-1', '-dpng');
    hold off;

    nus = 0.1:0.01:0.59;
    acc = zeros(length(nus), 1);
    % x = x(randperm(m), :);
    for i = 1:length(nus),
        libsvm_options = strcat(['-s 1 -t 0 -v 4 -n ' num2str(nus(i))]);
        acc(i) = svmtrain(c, x, libsvm_options);
    end

    plot(nus, acc);
    title('nu vs. accuracy');
    xlabel('nu');
    ylabel('acc');
    print('4-2', '-dpng');

    % Create a 20x20 grid that covers the training set
    coors_x = linspace(min(x(:, 1)), max(x(:, 1)), GRID_LEN);
    coors_y = linspace(min(x(:, 2)), max(x(:, 2)), GRID_LEN);
    [X, Y] = meshgrid(coors_x, coors_y);
    y = [reshape(X, [GRID_LEN^2 1]) reshape(Y, [GRID_LEN^2 1])];

    [~, I] = max(acc);
    libsvm_options = strcat(['-s 1 -t 0 -n ' num2str(0.09 + 0.01 * I)]);
    Md = svmtrain(c, x, libsvm_options);
    chat = svmpredict(ones(GRID_LEN^2, 1), y, Md);
    scatter(y(:, 1), y(:, 2), 40, chat);
    title('Grid colored by classified class labels');
    print('4-3', '-dpng');

    close all;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % RBF Kernel SVM data
    data = load('target.mat');
    c = data.c;
    x = data.x;
    [m, n] = size(x);

    figure;
    hold all;
    scatter(x(c==1, 1), x(c==1, 2), 'r');
    scatter(x(c==-1, 1), x(c==-1, 2), 'b');
    title('TARGET data points colored by their classes');
    print('4-4', '-dpng');
    hold off;

    gamma = 2.^(-15:15);
    acc = zeros(length(gamma), 1);
    for i = 1:length(gamma),
        libsvm_options = strcat(['-s 1 -t 2 -n 0.5 -v 4 -g ' num2str(gamma(i))]);
        acc(i) = svmtrain(c, x, libsvm_options);
    end

    plot(gamma, acc);
    title('gamma vs. accuracy');
    xlabel('gamma');
    ylabel('acc');
    print('4-5', '-dpng');

    % Create a 30x30 grid that covers the training set
    coors_x = linspace(min(x(:, 1)), max(x(:, 1)), GRID_LEN);
    coors_y = linspace(min(x(:, 2)), max(x(:, 2)), GRID_LEN);
    [X, Y] = meshgrid(coors_x, coors_y);
    y = [reshape(X, [GRID_LEN^2 1]) reshape(Y, [GRID_LEN^2 1])];

    [~, I] = max(acc);
    libsvm_options = strcat(['-s 1 -t 2 -n 0.5 -g ' num2str(gamma(I))]);
    Md = svmtrain(c, x, libsvm_options);
    chat = svmpredict(ones(GRID_LEN^2, 1), y, Md);
    scatter(y(:, 1), y(:, 2), 40, chat);
    title('Grid colored by classified class labels');
    print('4-6', '-dpng');

    close all;

return
end
