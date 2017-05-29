import numpy as np
import npcompare as npc
import scipy.stats as stats
from multiprocessing import Pool

import unittest
import numpy as np
import pystan

class LindleySameBFS(unittest.TestCase):
    def test_lindley_same_bfs(self):
        obs1 = np.random.normal(0, 2, size=100)
        obs2 = np.random.normal(0.4, 2, size=90)
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

        self.assertTrue((np.absolute(lindley.bfs1.beta.mean(axis=0)
            - bfs1.beta.mean(axis=0)) < 1e-03).all())
        self.assertTrue((np.absolute(lindley.bfs2.beta.mean(axis=0)
            - bfs2.beta.mean(axis=0)) < 1e-03).all())
        self.assertTrue((np.absolute(lindley.bfsconcat.beta.mean(axis=0)
            - bfsconcat.beta.mean(axis=0)) < 1e-03).all())
