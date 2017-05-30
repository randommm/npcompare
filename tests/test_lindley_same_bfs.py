import numpy as np
import npcompare as npc
import scipy.stats as stats
from multiprocessing import Pool

import unittest
import numpy as np
import pystan

class LindleySameBFS(unittest.TestCase):
    def test_1(self):
        obs1 = np.random.normal(0, 2, size=60)
        obs2 = np.random.normal(0.05, 2, size=70)
        obsconcat = np.hstack((obs1, obs2))

        lindley = npc.EstimateLindleyBFS(obs1, obs2, nmaxcomp=4,
                                         mixture=False,
                                         transformation="logit")
        lindley.sampleposterior(niter=100000, nchains=2)


        bfs1 = npc.EstimateBFS(obs1, nmaxcomp=4,
                               mixture=False,
                               transformation="logit")
        bfs1.sampleposterior(niter=100000, nchains=2)


        bfs2 = npc.EstimateBFS(obs2, nmaxcomp=4,
                               mixture=False,
                               transformation="logit")
        bfs2.sampleposterior(niter=100000, nchains=2)


        bfsconcat = npc.EstimateBFS(obsconcat, nmaxcomp=4,
                                    mixture=False,
                                    transformation="logit")
        bfsconcat.sampleposterior(niter=100000, nchains=2)


        lindley.evalgrid()
        bfs1.evalgrid()
        bfs2.evalgrid()

        obj1 = lindley.bfs1.egresults['logdensitymean']
        obj2 = lindley.bfs2.egresults['logdensitymean']
        obj3 = bfs1.egresults['logdensitymean']
        obj4 = bfs2.egresults['logdensitymean']

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
