#----------------------------------------------------------------------
# Copyright 2017 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.    If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
from npcompare.fourierseries import fourierseries
from npcompare.estimatebfs import EstimateBFS
from scipy.integrate import quad

class Compare:
    """
    Compare two samples.

    Parameters
    ----------
    f1 : function
        Density function for the first population.
    f2 : function
        Density function for the second population.
    psamples1 : array like
        Must be either a one dimensional numpy.array
        (in which case, each element will be passed one at a time to f1)
        or a two dimensional numpy.array (in which case, each row will
        be passed one at a time to f1).
    psamples2 : array like
        Analog to psamples1.
    weights1 : array like
        Give weights to posterior psamples1.
        Set to None if each posterior sample has the same weight
        (the usual case for MCMC methods).
    weights2 : array like
        Analog to weights2.
    metric : function
        Metric function to be used.

        Defaults to:
        ::

           def metric(f1, f2, param1, param2): return quad(lambda x:
           (f1(x, param1) - f2(x, param2))**2, a, b)[0]

        Can be set to a user-defined function of the same signature.
    lower : float
        Lower integration limit passed to default metric function.
    upper : float
        Upper integration limit passed to default metric function.
    """
    def __init__(self, f1, f2, psamples1, psamples2, lower=0, upper=1,
                             weights1=None, weights2=None, metric=None):
        self.f1 = f1
        self.f2 = f2
        self.psamples1 = np.array(psamples1)
        self.psamples2 = np.array(psamples2)

        if weights1 is not None:
            self.weights1 = np.array(weights1)
            self.weights1 /= self.weights1.sum()
        else:
            self.weights1 = None
        if weights2 is not None:
            self.weights2 = np.array(weights2)
            self.weights2 /= self.weights2.sum()
        else:
            self.weights2 = None

        if metric is None:
            self.metric = self._int_squared_error
        else:
            self.metric = metric

        self.lower = lower
        self.upper = upper

        self.msamples = np.array([], dtype=np.float64)

    def _int_squared_error(self, f1, f2, param1, param2):
        return quad(lambda x: (f1(x, param1) - f2(x, param2))**2,
                               self.lower, self.upper, limit=1000)[0]

    @classmethod
    def frombfs(cls, bfsobj1, bfsobj2, transformation=True,
                metric=None):
        """
        Create a class Compare from two EstimateBFS objects.

        Parameters
        ----------
        bfsobj1 : EstimateBFS object
        bfsobj2 : EstimateBFS object
        transformation :
            If set to False, metric will be evaluated in [0, 1] without
            transformation (EstimateBFS class documentation).
            Otherwise, transformation will be applied.

            Ignored if bfsobj1 has no transformation (which implies
            sample space of [0, 1]).
        metric : function
            Metric function to be used, defaults to class's default.

        Returns
        -------
        New instance of class Compare
        """
        if not bfsobj1._ismixture:
            raise Exception("Only mixtures supported for now.")
        totalsim1 = bfsobj1.beta.shape[0] * bfsobj1.nmaxcomp
        totalsim2 = bfsobj2.beta.shape[0] * bfsobj2.nmaxcomp
        psamples1 = np.empty((totalsim1, bfsobj1.nmaxcomp + 1))
        psamples2 = np.empty((totalsim2, bfsobj2.nmaxcomp + 1))
        weights1 = np.empty(totalsim1)
        weights2 = np.empty(totalsim2)

        for (bfsobj, psamples, weights) in (
            [(bfsobj1, psamples1, weights1),
             (bfsobj2, psamples2, weights2)]):
            for i in range(bfsobj.beta.shape[0]):
                for k in range(bfsobj.nmaxcomp):
                    wl = i * bfsobj.nmaxcomp + k
                    psamples[wl, 0] = bfsobj.lognormconst[i, k]
                    psamples[wl, 1:(k+2)] = bfsobj.beta[i, 0:(k+1), k]
                    psamples[wl, (k+2):] = 0.0
                    weights[wl] = bfsobj.weights[i, k]

        self = cls(None, None, psamples1, psamples2, None, None,
                                     weights1, weights2, metric)

        if transformation and bfsobj1.transformation is not None:
            self.bfs1_transformation = bfsobj1.transformation
            self.bfs2_transformation = bfsobj2.transformation
            self._set_bfs_transformation()
            self.lower = bfsobj1.transf(0)
            self.upper = bfsobj1.transf(1)
            if (self.lower != bfsobj2.transf(0) or
                self.upper != bfsobj2.transf(1)):
                raise ValueError("bfsobj1 and bfsobj2 have different "
                                 "sample spaces.")
        else:
            self.bfsobj1_laditransf = self._dummy
            self.bfsobj2_laditransf = self._dummy
            self.lower = 0
            self.upper = 1

        self.f1 = self._bfs_f1
        self.f2 = self._bfs_f2
        self.bfsobj1_nmaxcomp = bfsobj1.nmaxcomp
        self.bfsobj2_nmaxcomp = bfsobj2.nmaxcomp

        return self

    def _dummy(self, x):
        return 0

    def _bfs_f1(self, x, psample):
        logd = (fourierseries(x, self.bfsobj1_nmaxcomp) *
            psample[1:]).sum() - psample[0]
        logd += self.bfsobj1_laditransf(x)
        return np.exp(logd)

    def _bfs_f2(self, x, psample):
        logd = (fourierseries(x, self.bfsobj2_nmaxcomp) *
            psample[1:]).sum() - psample[0]
        logd += self.bfsobj2_laditransf(x)
        return np.exp(logd)

    def _set_bfs_transformation(self):
        bfsobj1 = EstimateBFS(transformation=self.bfs1_transformation)
        self.bfsobj1_laditransf = bfsobj1.laditransf

        bfsobj2 = EstimateBFS(transformation=self.bfs2_transformation)
        self.bfsobj2_laditransf = bfsobj2.laditransf

    def __len__(self):
        return self.msamples.size

    def sampleposterior(self, niter=1000, refresh=100):
        """
        Compare two samples.

        Parameters
        ----------
        niter : integer
            Number of simulations to be draw.
        refresh : integer
            Interval of samples to print the amount of
            samples obtained so far.
            Set to 0 to disable printing.

        Returns
        -------
        self
        """
        result = np.empty(niter)

        psamples1_index = \
            np.random.choice(np.arange(self.psamples1.shape[0]),
            niter, p=self.weights1)
        psamples2_index = \
            np.random.choice(np.arange(self.psamples2.shape[0]),
            niter, p=self.weights2)

        psamples1 = self.psamples1[psamples1_index]
        psamples2 = self.psamples2[psamples2_index]
        for i in range(niter):
            result[i] = self.metric(self.f1, self.f2,
                                    psamples1[i], psamples2[i])
            if (refresh):
                if (not i%refresh):
                    print(i, "samples generated")

        self.msamples = np.hstack([self.msamples, result])

        return self

    def plot(self, ax=None, **kwargs):
        """
        Plot empirical CDF of metric samples.

        Parameters
        ----------
        ax : matplotlib axes
            Axis to plot, defaults to axes of a new figure.
        show : bool
            If True, calls matplotlib.pyplot plt.show() at end.
        **kwargs :
            Aditional named arguments passed to
            matplotlib.axes.Axes.step.

        Returns
        -------
        matplotlib.axes.Axes object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError('matplotlib package required to plot')
        if self.msamples.size == 0:
            return "No metric samples to plot"
        smsamples = np.sort(self.msamples)
        if ax is None:
            ax = plt.figure().add_subplot(111)
        ax.step(smsamples,
                np.arange(self.msamples.size) / self.msamples.size,
                **kwargs)
        return ax

    def __getstate__(self):
        d = self.__dict__.copy()

        #Remove objects that might have nested functions
        #They will be reconstructed once loading
        if "bfs1_transformation" in d.keys():
            del(d["bfsobj1_laditransf"])
            del(d["bfsobj2_laditransf"])
        return d

    def __setstate__(self, d):
        self.__dict__ = d

        #Reconstruct transformation functions
        if "bfs1_transformation" in d.keys():
            self._set_bfs_transformation()
