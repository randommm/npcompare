import numpy as np
import npcompare as npc
import scipy.stats as stats
from multiprocessing import Pool

import unittest
import numpy as np
import pystan

class BFSMixtureSingle(unittest.TestCase):
    def test_1(self):
        obs = np.random.normal(0, 1, size=100)

        mix_bfs = npc.EstimateBFS(obs, nmaxcomp=7,
                                  mixture=True,
                                  transformation="logit")
        mix_bfs.sampleposterior(niter=50000, nchains=2)

        sin_bfs3 = npc.EstimateBFS(obs, nmaxcomp=7,
                                   mixture=False,
                                   transformation="logit")
        sin_bfs3.sampleposterior(niter=50000, nchains=2)

        sin_bfs4 = npc.EstimateBFS(obs, nmaxcomp=6,
                                   mixture=False,
                                   transformation="logit")
        sin_bfs4.sampleposterior(niter=50000, nchains=2)

        mix_bfs.evalgrid()
        sin_bfs3.evalgrid()
        sin_bfs4.evalgrid()

        obj1 = mix_bfs.egresults['logdensityindivmean'][6]
        obj2 = mix_bfs.egresults['logdensityindivmean'][5]
        obj3 = sin_bfs3.egresults['logdensitymean']
        obj4 = sin_bfs4.egresults['logdensitymean']

        abs_dist = np.absolute(obj1 - obj3)
        print("abs_dist: ")
        print(abs_dist)

        abs_mean = np.absolute(obj1 + obj3) / 2
        rel_dist = abs_dist / abs_mean
        print("rel_dist: ")
        print(rel_dist)

        self.assertLess(rel_dist.max(), .04)
        self.assertLess(rel_dist.mean(), .02)
        self.assertLess(np.median(rel_dist), .02)

        abs_dist = np.absolute(obj2 - obj4)
        print("abs_dist: ")
        print(abs_dist)

        abs_mean = np.absolute(obj2 + obj4) / 2
        rel_dist = abs_dist / abs_mean
        print("rel_dist: ")
        print(rel_dist)

        self.assertLess(rel_dist.max(), .04)
        self.assertLess(rel_dist.mean(), .02)
        self.assertLess(np.median(rel_dist), .02)
