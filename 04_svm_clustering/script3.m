function script3
   
    % Linear SVM testing dat 
    data = load('simple_iris.mat');
    x = data.x;
    c = data.c;
    kh = @(x,y) x*y';
    Md = svm_train(c,x,kh);
    [chat, d] = svm_classify(Md, x);

    figure;
    hold all;
    plot(x(c==1, 1), x(c==1, 2), 'bo');
    plot(x(c==-1, 1), x(c==-1, 2), 'ro');
    scatter(x(abs(Md.v)>0.001, 1), x(abs(Md.v)>0.001, 2), 50, 'k', 'filled');
    hold off;

    % Quadratic kernel SVM testing data
    data = load('simple_nonlinear.mat');
    % x = data.x;
    % c = data.c;
    % kh = []; % implement quadratic kernel
    % Md = svm_train(c,x,kh);

return
end

function Md = svm_train(c,x,kh)
    Md = [];
    N = size(c, 1);
    H = zeros(N, N);
    for i = 1:N,
        for j = 1:N,
            H(i, j) = c(i) * c(j) * kh(x(i, :), x(j, :));
        end
    end

    f = - ones(N, 1);
    A = [];
    b = [];
    Aeq = c';
    beq = 0;
    lb = zeros(N, 1);
    ub = inf(N, 1);

    lambda = quadprog(H, f, A, b, Aeq, beq, lb, ub);

    bias = 0;
    non_zeros = 0;
    for j = 1:N,
        if lambda(j) > 0,
            non_zeros = non_zeros + 1;
            bias = bias - c(j);
            for i = 1:N,
                bias = bias + c(i) * lambda(i) * kh(x(i, :), x(j, :));
            end
        end
    end

    v = zeros(N, 1);
    for i = 1:N,
        v(i) = c(i) * lambda(i);
    end

    Md.b = bias / non_zeros;
    Md.v = v;
    Md.x = x;
    Md.kh = kh;
    Md.N = N;

return
end


function [chat,d] = svm_classify(Md,y)
    l = size(y, 1);
    chat = zeros(l, 1);
    d = zeros(l, 1) - Md.b;
    N = Md.N;

    for row = 1:l,
        for i = 1:N,
            d(row) = d(row) + Md.v(i) * Md.kh(y(row, :), Md.x(i, :));
        end

        if d(row) > 0,
            chat(row) = 1;
        else,
            chat(row) = -1;
        end
    end
return
end








