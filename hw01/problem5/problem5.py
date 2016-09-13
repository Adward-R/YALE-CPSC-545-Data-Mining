__author__ = 'Adward'
import numpy as np
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt

plt.xlabel('Dist')
plt.ylabel('Count')
for k in range(1, 4):
    Xk = np.random.randn(1000, k + 1)
    radius = np.linalg.norm(Xk, ord=2, axis=1)
    for i in range(1000):
        Xk[i] = np.divide(Xk[i], radius[i])

    Dk = pdist(Xk, 'euclidean')
    plt.xlim([min(Dk), max(Dk)])

    bins = np.arange(min(Dk), max(Dk), (max(Dk)-min(Dk))/25)  # fixed bin size
    plt.hist(Dk, bins=bins, alpha=0.5)
    plt.title('Problem 3: eqwidth_' + str(k))
    plt.savefig('eqwidth_' + str(k) + '.png')
    plt.close()

    Dk.sort()
    bins = Dk[np.arange(0, len(Dk)-1, len(Dk)//25)]
    plt.hist(Dk, bins=bins, alpha=0.5)
    plt.title('Problem 3: eqpoints_' + str(k))
    plt.savefig('eqpoints_' + str(k) + '.png')
    plt.close()
