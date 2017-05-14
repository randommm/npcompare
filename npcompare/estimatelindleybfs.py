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
    obs1 : array like
        List/tuple or numpy.array 1D array of observations for model 1.
    obs2 : array like
        List/tuple or numpy.array 1D array of observations for model 2.
    hplindley : float
        A priori probability that obs1 and obs2 came from different
        populations.
    **kwargs:
        Additional keyword arguments passed to method `fit`.

    Notes
    ----------
    For other parameters, see documentation for EstimateBFS.
    """
    def __init__(self, obs1=None, obs2=None, nmaxcomp=10, hpp=1,
                 hpgamma=0, hplindley=.5, transformation=None,
                 mixture=True, **kwargs):
        self._ismixture = mixture
        self._smodel = None
        self.__isinitialized = False
        if not "niter" in kwargs:
            kwargs["niter"] = 0
        self.fit(obs1, obs2, nmaxcomp, hpp, hpgamma, hplindley,
                 transformation, **kwargs)

    def fit(self, obs1, obs2, nmaxcomp=None, hpp=None, hpgamma=None,
            hplindley=None, transformation=None, niter=5000, **kwargs):
        """
        Configure object model data and automatically calls method
        `sampleposterior` (to obtain MCMC samples from the posterior).

        Parameters
        ----------
        obs1 : array like
            List/tuple or numpy.array 1D array of observations for model
            1.
        obs2 : array like
            List/tuple or numpy.array 1D array of observations for model
            2.
        hplindley : float
            A priori probability that obs1 and obs2 came from different
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

        if obs1 is None or obs2 is None:
            obsconcat = None
        else:
            obsconcat = np.hstack((obs1, obs2))

        if hplindley is not None or not self.__isinitialized:
            self.hplindley = hplindley

        if self.__isinitialized:
            self.bfs1.fit(obs1, nmaxcomp, hpp, hpgamma,
                          transformation, 0, **kwargs)
            self.bfs2.fit(obs2, nmaxcomp, hpp, hpgamma,
                          transformation, 0, **kwargs)
            self.bfsconcat.fit(obsconcat, nmaxcomp, hpp, hpgamma,
                               transformation, **kwargs)
        else:
            self.bfs1 = EstimateBFS(obs1, nmaxcomp, hpp, hpgamma,
                                    transformation, self._ismixture,
                                    **kwargs)
            self.bfs2 = EstimateBFS(obs2, nmaxcomp, hpp, hpgamma,
                                    transformation, self._ismixture,
                                    **kwargs)
            self.bfsconcat = EstimateBFS(obsconcat, nmaxcomp, hpp,
                                         hpgamma, transformation,
                                         self._ismixture, **kwargs)
            self.__isinitialized = True


        if self.bfsconcat.modeldata is not None:
            self.modeldata = dict(self.bfsconcat.modeldata)
            self.modeldata["hplindley"] = self.hplindley
            self.modeldata["nobs1"] = self.bfs1.nobs
            self.modeldata["nobs2"] = self.bfs2.nobs
            self.modeldata["phi1"] = self.bfs1.phi
            self.modeldata["phi2"] = self.bfs2.phi
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
  .StanModel(model_code = EstimateLindleyBFS._smodelcode_mixture))
                self._smodel = EstimateLindleyBFS._smodel_mixture
            else:
                if EstimateLindleyBFS._smodel_single is None:
                    EstimateLindleyBFS._smodel_single = (pystan
  .StanModel(model_code=EstimateLindleyBFS._smodelcode_single))
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
            niter += niter / 10
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
        self.bfs1.sfit = self.bfs2.sfit = self.sfit
        self.bfsconcat.sfit = self.sfit
        self.bfs1._processfit(beta="beta1",
                              lognormconst="lognormconst1",
                              weights="weights1")
        self.bfs2._processfit(beta="beta2",
                              lognormconst="lognormconst2",
                              weights="weights2")
        self.bfsconcat._processfit(beta="betaconcat",
                                   lognormconst="lognormconstconcat",
                                   weights="weightsconcat")
        self.nsim = self.bfs1.nsim

        weightsfull = self.sfit.extract("weightsfull")["weightsfull"]
        self.problindley = weightsfull.mean(0)
        if self._ismixture:
            self.bfs1.weights *= weightsfull[:, 0, None]
            self.bfs2.weights *= weightsfull[:, 0, None]
            self.bfsconcat.weights *= weightsfull[:, 1, None]
        else:
            self.bfs1.weights = np.array(weightsfull[:, 0])
            self.bfs2.weights = self.bfs1.weights
            self.bfsconcat.weights = np.array(weightsfull[:, 1])

    def evalgrid(self, gridsize=1000):
        """
        Call method evalgrid for each EstimateBFS object that the self
        holds, that is, this is equivalent to calling:
        ::

            obj.bfs1.evalgrid(gridsize)
            obj.bfs2.evalgrid(gridsize)
            obj.bfsconcat.evalgrid(gridsize)

        Parameters
        ----------
        gridsize : size of the grid.

        Returns
        -------
        self
        """
        self.bfs1.evalgrid(gridsize)
        self.bfs2.evalgrid(gridsize)
        self.bfsconcat.evalgrid(gridsize)

        return self

    def __getstate__(self):
        d = OrderedDict(self.__dict__)
        #Ensure that self._smodel will be pickled first
        d.move_to_end("sfit")
        d.move_to_end("bfs1")
        d.move_to_end("bfs2")
        d.move_to_end("bfsconcat")
        return d

    _smodel_mixture = None
    _smodel_single = None

    _smodelcode_mixture = """
functions {
  real lkk(real x, real[] beta) {
    real result_;
    result_ = 0;
    for (i in 1:num_elements(beta)) {
      if (i % 2 == 0)
        result_ = result_ + beta[i] * cos(i * pi() * x);
      else
        result_ = result_ + beta[i] * sin((i + 1) * pi() * x);
    }

    return exp(result_ * sqrt2());
  }
  real[] integ(real x, real[] f, real[] beta, real[] x_r, int[] x_i) {
    real dfdx[1];
    dfdx[1] = lkk(x, beta);
    return dfdx;
  }
}
data {
  int<lower=1> nmaxcomp; // number of mixture components
  int<lower=1> nobs1; // number of data points
  int<lower=1> nobs2; // number of data points

  matrix[nobs1, nmaxcomp] phi1;
  matrix[nobs2, nmaxcomp] phi2;
  real<lower=0> hpp;
  real<lower=0> hpgamma;

  real<lower=0> hplindley;
}
transformed data {
  real x_r[0];
  int x_i[0];
  matrix[nobs1 + nobs2, nmaxcomp] phiconcat;
  real minus_hpp_minus_half;
  int<lower=1> nobsconcat;
  real lhplindley[2];
  real minus_hpgamma_times_i[nmaxcomp];

  phiconcat = append_row(phi1, phi2);

  minus_hpp_minus_half = -hpp - 0.5;

  nobsconcat = nobs1 + nobs2;
  lhplindley[1] = log(hplindley);
  lhplindley[2] = log1m(hplindley);

  for (i in 1:nmaxcomp)
    minus_hpgamma_times_i[i] = -hpgamma * i;
}
parameters {
  matrix[nmaxcomp, nmaxcomp] beta1;
  matrix[nmaxcomp, nmaxcomp] beta2;
  matrix[nmaxcomp, nmaxcomp] betaconcat;
}
transformed parameters {
  vector[2] lpfull;
  vector[nmaxcomp] lp1;
  vector[nmaxcomp] lp2;
  vector[nmaxcomp] lpconcat;
  real lognormconst1[nmaxcomp];
  real lognormconst2[nmaxcomp];
  real lognormconstconcat[nmaxcomp];
  for (i in 1:nmaxcomp) {
    lognormconst1[i] =
      log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                             to_array_1d(beta1[1:i, i]),
                             x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
    lp1[i] = sum(phi1[, 1:i] * beta1[1:i, i])
             - nobs1 * lognormconst1[i]
             + minus_hpgamma_times_i[i];

    lognormconst2[i] =
      log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                             to_array_1d(beta2[1:i, i]),
                             x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
    lp2[i] = sum(phi2[, 1:i] * beta2[1:i, i])
             - nobs2 * lognormconst2[i]
             + minus_hpgamma_times_i[i];

    lognormconstconcat[i] =
      log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                             to_array_1d(betaconcat[1:i, i]),
                             x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
    lpconcat[i] = sum(phiconcat[, 1:i] * betaconcat[1:i, i])
                  - nobsconcat * lognormconstconcat[i]
                  + minus_hpgamma_times_i[i];
  }

  lpfull[1] = log_sum_exp(lp1) + log_sum_exp(lp2) + lhplindley[1];
  lpfull[2] = log_sum_exp(lpconcat) + lhplindley[2];
}
model {
  target += log_sum_exp(lpfull[1], lpfull[2]);
  for (i in 1:nmaxcomp) {
    if (i % 2 == 0) {
      beta1[i, ] ~ normal(0, i ^ minus_hpp_minus_half);
      beta2[i, ] ~ normal(0, i ^ minus_hpp_minus_half);
      betaconcat[i, ] ~ normal(0, i ^ minus_hpp_minus_half);
    } else {
      beta1[i, ] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
      beta2[i, ] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
      betaconcat[i, ] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
    }
  }
}
generated quantities {
  int<lower=1,upper=nmaxcomp> model_index1;
  int<lower=1,upper=nmaxcomp> model_index2;
  int<lower=1,upper=nmaxcomp> model_indexconcat;
  int<lower=1,upper=2> model_indexfull;
  vector[nmaxcomp] weights1;
  vector[nmaxcomp] weights2;
  vector[nmaxcomp] weightsconcat;
  vector[2] weightsfull;

  weights1 = softmax(lp1);
  model_index1 = categorical_rng(weights1);
  weights2 = softmax(lp2);
  model_index2 = categorical_rng(weights2);
  weightsconcat = softmax(lpconcat);
  model_indexconcat = categorical_rng(weightsconcat);
  weightsfull = softmax(lpfull);
  model_indexfull = categorical_rng(weightsfull);
}
"""

    _smodelcode_single = """
functions {
  real lkk(real x, real[] beta) {
    real result_;
    result_ = 0;
    for (i in 1:num_elements(beta)) {
      if (i % 2 == 0)
        result_ = result_ + beta[i] * cos(i * pi() * x);
      else
        result_ = result_ + beta[i] * sin((i + 1) * pi() * x);
    }

    return exp(result_ * sqrt2());
  }
  real[] integ(real x, real[] f, real[] beta, real[] x_r, int[] x_i) {
    real dfdx[1];
    dfdx[1] = lkk(x, beta);
    return dfdx;
  }
}
data {
  int<lower=1> nmaxcomp; // number of mixture components
  int<lower=1> nobs1; // number of data points
  int<lower=1> nobs2; // number of data points

  matrix[nobs1, nmaxcomp] phi1;
  matrix[nobs2, nmaxcomp] phi2;
  real<lower=0> hpp;

  real<lower=0> hplindley;
}
transformed data {
  real x_r[0];
  int x_i[0];
  matrix[nobs1 + nobs2, nmaxcomp] phiconcat;
  real minus_hpp_minus_half;
  int<lower=1> nobsconcat;
  real lhplindley[2];

  phiconcat = append_row(phi1, phi2);

  minus_hpp_minus_half = -hpp - 0.5;

  nobsconcat = nobs1 + nobs2;
  lhplindley[1] = log(hplindley);
  lhplindley[2] = log1m(hplindley);
}
parameters {
  vector[nmaxcomp] beta1;
  vector[nmaxcomp] beta2;
  vector[nmaxcomp] betaconcat;
}
transformed parameters {
  vector[2] lpfull;
  real lognormconst1;
  real lognormconst2;
  real lognormconstconcat;

  lognormconst1 =
    log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                           to_array_1d(beta1),
                           x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);

  lognormconst2 =
    log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                           to_array_1d(beta2),
                           x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);

  lognormconstconcat =
    log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                           to_array_1d(betaconcat),
                           x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);

  lpfull[1] = sum(phi1 * beta1) - nobs1 * lognormconst1
              + sum(phi2 * beta2) - nobs2 * lognormconst2
              + lhplindley[1];

  lpfull[2] = sum(phiconcat * betaconcat)
              - nobsconcat * lognormconstconcat
              + lhplindley[2];
}
model {
  target += log_sum_exp(lpfull[1], lpfull[2]);
  for (i in 1:nmaxcomp) {
    if (i % 2 == 0) {
      beta1[i] ~ normal(0, i ^ minus_hpp_minus_half);
      beta2[i] ~ normal(0, i ^ minus_hpp_minus_half);
      betaconcat[i] ~ normal(0, i ^ minus_hpp_minus_half);
    } else {
      beta1[i] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
      beta2[i] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
      betaconcat[i] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
    }
  }
}
generated quantities {
  int<lower=1,upper=2> model_indexfull;
  vector[2] weightsfull;

  weightsfull = softmax(lpfull);
  model_indexfull = categorical_rng(weightsfull);
}
"""
