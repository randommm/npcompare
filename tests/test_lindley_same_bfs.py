import numpy as np
import npcompare as npc
import scipy.stats as stats
from multiprocessing import Pool

import unittest
import numpy as np
import pystan

class LindleySameBFS(unittest.TestCase):
    def test_1(self):
        obs0 = np.random.normal(0, 2, size=60)
        obs1 = np.random.normal(0.05, 2, size=70)
        obsconcat = np.hstack((obs0, obs1))

        lindley = npc.EstimateLindleyBFS(obs0, obs1, nmaxcomp=4,
                                         mixture=False,
                                         transformation="logit")
        lindley.sampleposterior(niter=100000, nchains=2)


        bfs0 = npc.EstimateBFS(obs0, nmaxcomp=4,
                               mixture=False,
                               transformation="logit")
        bfs0.sampleposterior(niter=100000, nchains=2)


        bfs1 = npc.EstimateBFS(obs1, nmaxcomp=4,
                               mixture=False,
                               transformation="logit")
        bfs1.sampleposterior(niter=100000, nchains=2)


        bfsconcat = npc.EstimateBFS(obsconcat, nmaxcomp=4,
                                    mixture=False,
                                    transformation="logit")
        bfsconcat.sampleposterior(niter=100000, nchains=2)


        lindley.evalgrid()
        bfs0.evalgrid()
        bfs1.evalgrid()

        obj0 = lindley.bfs0.egresults['logdensitymean']
        obj1 = lindley.bfs1.egresults['logdensitymean']
        obj2 = bfs0.egresults['logdensitymean']
        obj3 = bfs1.egresults['logdensitymean']

        abs_dist = np.absolute(obj0 - obj2)
        print("abs_dist: ")
        print(abs_dist)

        abs_mean = np.absolute(obj0 + obj2) / 2
        rel_dist = abs_dist / abs_mean
        print("rel_dist: ")
        print(rel_dist)

        self.assertLess(rel_dist.max(), .04)
        self.assertLess(rel_dist.mean(), .02)
        self.assertLess(np.median(rel_dist), .02)

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
