__author__ = 'Adward'

from numpy.random import randint
from numpy import log2, array


def entropy(v):  # v as 1-d vec in np.array
    m = len(v)
    err = 0
    for k in set(v):
        f = v[v == k].size / m
        err -= f * log2(f)
    return err


def gini(v):
    m = len(v)
    err = 1
    for k in set(v):
        f = v[v == k].size / m
        err -= f ** 2
    return err


# return info_gain obtained by selecting any column to divide
def info_gain(x, c, inpurity):
    igain = [inpurity(c)] * 2
    for j in range(2):
        for k in set(c):
            cj = c[x[:, j] == k]
            igain[j] -= (len(cj) / len(c)) * inpurity(cj)
    return igain


n_samples = 10
cnt, e_sel, g_sel = 0, 0, 0
while e_sel == g_sel:
    x = randint(2, size=(n_samples, 2))
    c = randint(2, size=n_samples)

    e_gain = info_gain(x, c, entropy)
    g_gain = info_gain(x, c, gini)

    e_sel = e_gain.index(max(e_gain))
    g_sel = g_gain.index(max(g_gain))

    cnt += 1

print(e_sel, g_sel, cnt)
print(e_gain, g_gain)
print(x)
print(c)