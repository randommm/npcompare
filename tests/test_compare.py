import numpy as np
import npcompare as npc
import unittest
import pystan
import random

import scipy.stats as stats
from scipy.integrate import quad

class Compare(unittest.TestCase):
    def test_1(self):
        np.random.seed(12)
        random.seed(12)

        obs0 = stats.beta.rvs(2, 3, size=150)
        obs1 = stats.beta.rvs(3, 2, size=150)

        kernel0 = stats.gaussian_kde(obs0)
        kernel1 = stats.gaussian_kde(obs1)
        kernel_dist = quad(lambda x: (kernel0.evaluate(x)
          - kernel1.evaluate(x))**2, 0, 1)[0]

        true_dist = quad(lambda x: (stats.beta.pdf(x, 2, 3)
          - stats.beta.pdf(x, 3, 2))**2, 0, 1)[0]

        bfs0 = npc.EstimateBFS(obs0, nmaxcomp=8,
                               mixture=True,
                               #transformation="logit"
                               )
        bfs0.sampleposterior(niter=10000, nchains=2)

        bfs1 = npc.EstimateBFS(obs1, nmaxcomp=8,
                               mixture=True,
                               #transformation="logit"
                               )
        bfs1.sampleposterior(niter=10000, nchains=2)

        comp = npc.Compare.frombfs(bfs0, bfs1)
        comp.sampleposterior(2000)
        bfs_dist = comp.msamples.mean()

        true_kde_dist = np.absolute(kernel_dist - true_dist)
        true_bfs_dist = np.absolute(bfs_dist - true_dist)

        print("true_kde_dist: ")
        print(true_kde_dist)

        print("true_bfs_dist: ")
        print(true_bfs_dist)

        self.assertLess(true_bfs_dist, true_kde_dist)
