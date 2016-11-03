#Script4   Complete the exercise only using the core Python programming language
#and the following imported functions as needed:

from scipy.io import loadmat as load
from numpy import argsort, reshape, array, log2, zeros, transpose


def script4():
    # all attr are categorical
    data = load('restaurant_1.mat')
    # [c] is an mx1 vector of int corrspd to the class labels for each of the m samples
    c = array(data['c']).astype(int)
    # [nc] is the number of classes, that is, c(i) belongs to {1,...,nc} for i = 1,...,m
    nc = array(data['nc']).astype(int)[0]
    # [x] is an mxn matrix of int corrspd to the n attr for each of the m training samples
    x = array(data['x']).astype(int)
    # [nx] is an 1xn vec corrspd to the n attr for each of the m training samples
    nx = array(data['nx']).astype(int)
    nx = reshape(nx, (1, nx.size))

    # [y] is an kxn matrix of int corrspd to the n attr for each of the k testing samples
    y = array(data['y']).astype(int)
    d = array(data['d']).astype(int)

    tr = tree_train(c, nc, x, nx)
    print(tr)
    b = tree_classify(y, tr)

    your_output = b
    correct_output = d

    return your_output, correct_output


def tree_train(c, nc, x, nx):
    nx = nx[0]
    prefix = [True] * len(nx)  # denote un-traversed columns
    c = c.transpose()[0]

    def entropy(v):  # v as 1-d vec in np.array
        m = len(v)
        err = 0
        for k in set(v):
            f = v[v == k].size / m
            err -= f * log2(f)
        return err

    def dtree_gen(_x, _c):  # _x includes part of samples & full variables
        if sum(prefix) == 1:  # leaf node, maybe with multi classes
            cnt = dict.fromkeys(set(_c), 0)
            for k in _c:
                cnt[k] += 1
            candidates = sorted(cnt, key=lambda ky: cnt[ky], reverse=True)
            res = candidates[0]
            c0 = cnt[res]
            if c0 == cnt[candidates[1]]:
                for k in candidates[1:]:
                    if cnt[k] != c0:
                        return res
                    elif k < res:
                        res = k
            return [res, None]
        elif len(set(_c)) == 1:  # leaf node, with only one class remaining
            return [_c[0], 0]

        max_info_gain = 0
        max_j = 0  # attr index at this step with max info_gain
        for j in range(len(nx)):
            if not prefix[j]:
                continue
            # calc info gain on this attr
            info_gain = entropy(_c)
            for k in range(nx[j]):
                cj = _c[_x[:, j] == k+1]
                try:
                    info_gain -= (len(cj) / len(_c)) * entropy(cj)
                except:
                    # return default label to bypass the divide-by-zero exception cause by empty node problem
                    return [_c[0], 0]
            # > rather than >= guarantees if multi attr has the same info_gain, the first will apply
            if info_gain > max_info_gain:
                max_j = j
                max_info_gain = info_gain

        prefix[max_j] = False
        node = [max_j, []]  # node[0] indicates (index of) variable that used to divide here
        # node[1] includes a list of the same structures, saying, sub-nodes of categorical classes
        for j in range(nx[max_j]):  # multi sub-nodes
            subclass = list(filter(lambda i: _x[i, max_j] == j+1, range(len(_x))))
            node[1].append(dtree_gen(_x[subclass, :], _c[subclass]))
        prefix[max_j] = True
        return node

    return dtree_gen(x, c)


def tree_classify(y,tr):
    b = []
    for i in range(len(y)):
        cls, subnode = tr
        while subnode:
            cls, subnode = subnode[y[i, cls]-1]
        b.append(cls)
    return reshape(array(b), (len(b), 1))


out = script4()
print(out)

