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

import pkg_resources
import numpy as np
from npcompare.estimatebfs import EstimateBFS
from npcompare.fourierseries import fourierseries
from collections import OrderedDict, Mapping
from scipy.special import logit, expit

class EstimateLindleyBFS:
    """
    Estimate univariate density using Bayesian Fourier Series.
    This method only works with data the lives in
    [0, 1], however, the class implements methods to automatically
    transform user inputted data to [0, 1]. See parameter `transform`
    below.

    Parameters
    ----------
    obs0 : array like
        List/tuple or numpy.array 1D array of observations for model 1.
    obs1 : array like
        List/tuple or numpy.array 1D array of observations for model 2.
    hplindley : float
        A priori probability that obs0 and obs1 came from different
        populations.
    **kwargs:
        Additional keyword arguments passed to method `fit`.

    Notes
    ----------
    For other parameters, see documentation for EstimateBFS.
    """
    def __init__(self, obs0=None, obs1=None, nmaxcomp=10, hpp=1,
                 hpgamma=0, hplindley=.5, transformation=None,
                 mixture=True, **kwargs):
        self._ismixture = mixture
        self._smodel = None
        self.__isinitialized = False
        if not "niter" in kwargs:
            kwargs["niter"] = 0
        self.fit(obs0, obs1, nmaxcomp, hpp, hpgamma, hplindley,
                 transformation, **kwargs)

    def fit(self, obs0, obs1, nmaxcomp=None, hpp=None, hpgamma=None,
            hplindley=None, transformation=None, niter=5000, **kwargs):
        """
        Configure object model data and automatically calls method
        `sampleposterior` (to obtain MCMC samples from the posterior).

        Parameters
        ----------
        obs0 : array like
            List/tuple or numpy.array 1D array of observations for model
            1.
        obs1 : array like
            List/tuple or numpy.array 1D array of observations for model
            2.
        hplindley : float
            A priori probability that obs0 and obs1 came from different
            populations.
        **kwargs:
            Additional keyword arguments passed to method `sample`.

        Returns
        -------
        self

        Notes
        ----------
        For other parameters, see documentation for method fit of
        EstimateBFS.
        """

        #Clean results
        self.sfit = None
        self.problindley = None
        self.nsim = None

        if obs0 is None or obs1 is None:
            obsconcat = None
        else:
            obsconcat = np.hstack((obs0, obs1))

        if hplindley is not None or not self.__isinitialized:
            self.hplindley = hplindley

        if self.__isinitialized:
            self.bfs0.fit(obs0, nmaxcomp, hpp, hpgamma,
                          transformation, 0, **kwargs)
            self.bfs1.fit(obs1, nmaxcomp, hpp, hpgamma,
                          transformation, 0, **kwargs)
            self.bfsconcat.fit(obsconcat, nmaxcomp, hpp, hpgamma,
                               transformation, **kwargs)
        else:
            self.bfs0 = EstimateBFS(obs0, nmaxcomp, hpp, hpgamma,
                                    transformation, self._ismixture,
                                    **kwargs)
            self.bfs1 = EstimateBFS(obs1, nmaxcomp, hpp, hpgamma,
                                    transformation, self._ismixture,
                                    **kwargs)
            self.bfsconcat = EstimateBFS(obsconcat, nmaxcomp, hpp,
                                         hpgamma, transformation,
                                         self._ismixture, **kwargs)
            self.__isinitialized = True


        if self.bfsconcat.modeldata is not None:
            self.modeldata = dict(self.bfsconcat.modeldata)
            self.modeldata["hplindley"] = self.hplindley
            self.modeldata["nobs0"] = self.bfs0.nobs
            self.modeldata["nobs1"] = self.bfs1.nobs
            self.modeldata["phi1"] = self.bfs0.phi
            self.modeldata["phi2"] = self.bfs1.phi
            del(self.modeldata["nobs"])
            del(self.modeldata["phi"])
        else:
            self.modeldata = None

        if niter != 0:
            self.sampleposterior(niter, **kwargs)

        return self

    def __len__(self):
         return self.beta.shape[0]

    def compilestanmodel(self):
        """
        Compile Stan model necessary for method sample. This method is
        called automatically by obj.sample().

        Returns
        -------
        self
        """
        try:
            import pystan
        except ImportError:
            raise ImportError('pystan package required for class '
                              'Estimate')
        if self._smodel is None:
            if self._ismixture:
                if EstimateLindleyBFS._smodel_mixture is None:
                    EstimateLindleyBFS._smodel_mixture = (pystan
  .StanModel(file = EstimateLindleyBFS._smodelcode_mixture))
                self._smodel = EstimateLindleyBFS._smodel_mixture
            else:
                if EstimateLindleyBFS._smodel_single is None:
                    EstimateLindleyBFS._smodel_single = (pystan
  .StanModel(file = EstimateLindleyBFS._smodelcode_single))
                self._smodel = EstimateLindleyBFS._smodel_single

        return self

    def sampleposterior(self, niter=5000, nchains=4, njobs=-1,
                                            tolrhat=0.02, **kwargs):
        """
        Samples from posterior.

        Parameters
        ----------
        niter :
            Number of simulations to be draw.
        nchains :
            Number of MCMC chains.
        njobs :
            Number of CPUs to be used in parallel. If -1 (default), all
            CPUs will be used.
        tolrhat :
            Maximum tolerable distance an Rhat of any sampled parameter
            can have from 1 (we will resample model approximatedly 10%
            more iterations until this convergence criteria is met).
        **kwargs :
            Additional keyword arguments passed to pystan (e.g.: refresh
            parameter to configure sampler printing status).

        Returns
        -------
        self
        """
        self.compilestanmodel()

        if self.modeldata is None:
            raise Exception('Data is not set, must call method fit '
                            'first.')

        while True:
            self.sfit = self._smodel.sampling(data=self.modeldata,
                                              n_jobs=njobs,
                                              chains=nchains,
                                              iter=int(niter), **kwargs)
            irhat = self.sfit.summary()['summary_colnames']
            irhat = irhat.index("Rhat")
            drhat1 = self.sfit.summary()["summary"][:, irhat]
            drhat1 = max(abs(drhat1 - 1))
            if drhat1 < tolrhat:
                break
            niter += niter / 2
            print("Model failed to converge given your tolrhat of ",
                  tolrhat,
                  "; the observed maximum distance of an Rhat from 1 "
                  "was ", drhat1,
                  "; retrying sampling with ", niter, "iterations")

        self._processfit()
        self.egresults = dict()

        return self

    def _processfit(self):
        if not self.sfit:
            raise Exception("This function cannot be called before "
                            "obj.sampleposterior()")
        self.bfs0.sfit = self.bfs1.sfit = self.sfit
        self.bfsconcat.sfit = self.sfit
        self.bfs0._processfit(beta="beta1",
                              lognormconst="lognormconst1",
                              weights="weights1")
        self.bfs1._processfit(beta="beta2",
                              lognormconst="lognormconst2",
                              weights="weights2")
        self.bfsconcat._processfit(beta="betaconcat",
                                   lognormconst="lognormconstconcat",
                                   weights="weightsconcat")
        self.nsim = self.bfs0.nsim

        weightsfull = self.sfit.extract("weightsfull")["weightsfull"]
        self.problindley = weightsfull.mean(0)
        if self._ismixture:
            self.bfs0.weights *= weightsfull[:, 0, None]
            self.bfs1.weights *= weightsfull[:, 0, None]
            self.bfsconcat.weights *= weightsfull[:, 1, None]
        else:
            self.bfs0.weights = np.array(weightsfull[:, 0])
            self.bfs1.weights = self.bfs0.weights
            self.bfsconcat.weights = np.array(weightsfull[:, 1])

    def evalgrid(self, gridsize=1000):
        """
        Call method evalgrid for each EstimateBFS object that the self
        holds, that is, this is equivalent to calling:
        ::

            obj.bfs0.evalgrid(gridsize)
            obj.bfs1.evalgrid(gridsize)
            obj.bfsconcat.evalgrid(gridsize)

        Parameters
        ----------
        gridsize : size of the grid.

        Returns
        -------
        self
        """
        self.bfs0.evalgrid(gridsize)
        self.bfs1.evalgrid(gridsize)
        self.bfsconcat.evalgrid(gridsize)

        return self

    def __getstate__(self):
        d = OrderedDict(self.__dict__)
        #Ensure that self._smodel will be pickled first
        d.move_to_end("sfit")
        d.move_to_end("bfs0")
        d.move_to_end("bfs1")
        d.move_to_end("bfsconcat")
        return d

    _smodel_mixture = None
    _smodel_single = None

    _smodelcode_mixture = (pkg_resources.resource_filename(__name__,
        "models/estimatelindleybfs_mixture.stan"))

    _smodelcode_single = (pkg_resources.resource_filename(__name__,
        "models/estimatelindleybfs_single.stan"))
