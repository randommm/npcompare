import numpy as np
import npcompare as npc
import unittest
import pystan
import random

class BFSMixtureSingle(unittest.TestCase):
    def test_1(self):
        np.random.seed(12)
        random.seed(12)

        obs = np.random.normal(0, 1, size=100)

        mix_bfs = npc.EstimateBFS(obs, nmaxcomp=7,
                                  mixture=True,
                                  transformation="logit")
        mix_bfs.sampleposterior(niter=50000, nchains=2)

        sin_bfs2 = npc.EstimateBFS(obs, nmaxcomp=7,
                                   mixture=False,
                                   transformation="logit")
        sin_bfs2.sampleposterior(niter=50000, nchains=2)

        sin_bfs3 = npc.EstimateBFS(obs, nmaxcomp=6,
                                   mixture=False,
                                   transformation="logit")
        sin_bfs3.sampleposterior(niter=50000, nchains=2)

        mix_bfs.evalgrid()
        sin_bfs2.evalgrid()
        sin_bfs3.evalgrid()

        obj0 = mix_bfs.egresults['logdensityindivmean'][6]
        obj1 = mix_bfs.egresults['logdensityindivmean'][5]
        obj2 = sin_bfs2.egresults['logdensitymean']
        obj3 = sin_bfs3.egresults['logdensitymean']

        abs_dist = np.absolute(obj0 - obj2)
        print("abs_dist: ")
        print(abs_dist)

        abs_mean = np.absolute(obj0 + obj2) / 2
        rel_dist = abs_dist / abs_mean
        print("rel_dist: ")
        print(rel_dist)

        self.assertLess(rel_dist.max(), .09)
        self.assertLess(rel_dist.mean(), .04)
        self.assertLess(np.median(rel_dist), .04)

        abs_dist = np.absolute(obj1 - obj3)
        print("abs_dist: ")
        print(abs_dist)

        abs_mean = np.absolute(obj1 + obj3) / 2
        rel_dist = abs_dist / abs_mean
        print("rel_dist: ")
        print(rel_dist)

        self.assertLess(rel_dist.max(), .09)
        self.assertLess(rel_dist.mean(), .04)
        self.assertLess(np.median(rel_dist), .04)
