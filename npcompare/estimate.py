#----------------------------------------------------------------------
# Copyright 2017 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
from npcompare.fourierseries import fourierseries
from collections import OrderedDict
import scipy.special

def logit(x):
  return - np.log(1 / x - 1)

def invlogit(x):
  return 1 / (1 + np.exp(-x))

class EstimateBFS:
  """
  Estimate univariate density using Bayesian Fourier Series
  with a sieve prior. This method only works with data the lives in
  [0, 1], however, the class implements methods to automatically
  transform user inputted data to [0, 1]. See parameter `transform`
  below.

  Parameters
  ----------
  obs : list/tuple or numpy.array 1D array of observations.
  nmaxcomp : maximum number of components of the Fourier series
    expansion.
  hpp : hiperparameter p (defaults to "conservative" value of 1).
  hpgamma : hiperparameter p (defaults to "conservative" value of 0).
  gridsize : size of the grid of variables for which we'll calculate the
    predicted values of the density (after posterior is finished
    sampling).
  transform : transformation function to use. Can be either:

    string "logit" to use logit transformation. Usefull if the sample
    space is the real line. However the `obs` must actually assume
    only low values like, for example, between [-5, 5], this is due to
    the fact that inverse logit starts getting (computationally) very
    close to 1 (or 0) after 10 (or -10).

    dictionary {"transf": "fixed", "vmin": vmin, "vmax": vmax} for fixed
    value transformation, where vmin and vmax are the minimun and
    maximum values of the sample space (for `obs`), respectivelly.

    A user defined transformation function with a dict with elements
    `transf`, `itransf` and `laditransf`, where `transf` is a function
    that transforms [0, 1] to sample space, `itransf` is its inverse,
    and `laditransf` is the log absolute derivative of `itransf`.
    These 3 functions must accept and return numpy 1D arrays.
  """
  def __init__(self, obs=None, nmaxcomp=10, hpp=1, hpgamma=0,
               gridsize=1000, transformation=None):
    self.obs = np.array(obs, ndmin=1)
    self._dp = np.linspace(0, 1, gridsize)

    if transformation:
      if transformation == "logit":
        self.transf = scipy.special.logit #logit
        self.itransf = scipy.special.expit #invlogit
        logoftwo = np.log(2)
        self.laditransf = \
          lambda x: - np.logaddexp(logoftwo, np.logaddexp(x, -x))
      elif isinstance(transformation, dict):
        if (transformation["transf"] == "fixed"):
          vmax = transformation["vmax"]
          vmin = transformation["vmin"]
          vmaxmvmin = vmax - vmin
          livmaxmvmin = -np.log(vmaxmvmin)
          self.transf = lambda x: vmaxmvmin * x + vmin
          self.itransf = lambda x: (x - vmin) / vmaxmvmin
          self.laditransf = lambda x: livmaxmvmin
        else:
          self.transf = transformation["transf"]
          self.itransf = transformation["itransf"]
          self.laditransf = transformation["laditransf"]
      else:
        Exception("Unrecognized parameter transformation")
      self.gridpoints = self.transf(self._dp)
      self.itobs = self.itransf(obs)
    else:
      self.gridpoints = self._dp
      self.itobs = self.obs

    self.transformation = transformation
    self.phi = fourierseries(self.itobs, nmaxcomp)
    self._phidp = fourierseries(self._dp, nmaxcomp)
    self.nmaxcomp = nmaxcomp
    self.nobs = self.obs.size
    self.hpp = hpp
    self.hpgamma = hpgamma
    self.modeldata = dict(nobs=self.nobs, phi=self.phi, hpp=hpp,
                          nmaxcomp=nmaxcomp, hpgamma=hpgamma)
    self.probcomp = None
    self.sfit = None

  def __len__(self):
     return self.beta.shape[0]

  def compilestanmodel(self):
    """
    Compile Stan model necessary for method sample. This method is
      called automatically by method sample.
    """
    try:
      import pystan
    except ImportError:
      raise ImportError('pystan package required for class Estimate')

    if not self._smodel:
      EstimateBFS._smodel = \
        pystan.StanModel(model_code=self._smodelcode)

  def sample(self, niter=1000, nchains=4, njobs=-1, tolrhat=0.02,
             **kwargs):
    """
    Samples from posterior.

    Parameters
    ----------
    niter : number of simulations to be draw.
    nchains : number of MCMC chains.
    njobs : number of CPUs to be used in parallel. If -1 (default),
      all CPUs will be used.
    tolrhat : maximum tolerable distance an Rhat of any sampled
      parameter can have from 1 (we will resample model approximatedly
      10% more iterations until this convergence criteria is met).
    **kwargs : aditional named arguments passed to pystan (e.g.:
      refresh parameter to configure sampler printing status).

    Returns
    -------
    None
    """
    self.compilestanmodel()
    self._smodel = EstimateBFS._smodel

    while True:
      self.sfit = self._smodel.sampling(data=self.modeldata, n_jobs=njobs,
                                        chains=nchains, iter=niter,
                                        **kwargs)
      irhat = self.sfit.summary()['summary_colnames'].index("Rhat")
      drhat1 = max(abs(self.sfit.summary()["summary"][:, irhat] - 1))
      if drhat1 < tolrhat:
        break
      niter += niter // 10
      print("Model failed to converge given your tolrhat of ", tolrhat,
            "; the observed maximum distance of an Rhat from 1 was ",
            drhat1, "; retrying sampling with ", niter, "iterations")

    self.__processfit()
    self.isgridevalued = False

  def __processfit(self):
    if not self.sfit:
      Exception("This function cannot be called before obj.sample()")

    #self.beta = self.sfit.extract("beta")["beta"]
    #self.lognormconst = \
      #self.sfit.extract("lognormconst")["lognormconst"]
    #self.weights = self.sfit.extract("weights")["weights"]
    #self.nsim = self.beta.shape[0]
    #self.probcomp = self.weights.mean(0)

    def beta_func(i, j, k):
      return extracted[i, fnames.index("beta["+str(j)+","+str(k)+"]")]
    def lognormconst_func(i, j):
      return extracted[i, fnames.index("lognormconst["+str(j)+"]")]
    def weights_func(i, j):
      return extracted[i, fnames.index("weights["+str(j)+"]")]

    def fromfunction2d(func, shape):
      res = np.empty(shape)
      for j in range(shape[1]):
        res[range(shape[0]), j] = func(range(shape[0]), j)
      return res

    def fromfunction3d(func, shape):
      res = np.empty(shape)
      for j in range(shape[1]):
        for k in range(shape[2]):
          res[range(shape[0]), j, k] = func(range(shape[0]), j, k)
      return res

    nmaxcomp = self.nmaxcomp
    fnames = self.sfit.flatnames
    extracted = self.sfit.extract(permuted=False)
    nsim = extracted.shape[0] * extracted.shape[1]
    extracted = extracted.reshape((nsim, len(fnames) + 1))
    self.beta = fromfunction3d(beta_func, (nsim, nmaxcomp, nmaxcomp))
    self.lognormconst = fromfunction2d(lognormconst_func,
                                       (nsim, nmaxcomp))
    self.weights = fromfunction2d(weights_func, (nsim, nmaxcomp))

    self.nsim = nsim
    self.probcomp = self.weights.mean(0)

  def evalgrid(self):
    """
    Calculates posterior values at grid points so they can be used
      later by methods like plot method or direct

    Returns
    -------
    None
    """
    self.__processfit()

    self.logposteriorindivmean =\
      np.empty((self.nmaxcomp, self._dp.size))
    for i in range(self.nmaxcomp):
      self.logposteriorindivmean[i, :] =\
        self.__predictdensityindiv(self.gridpoints, self._phidp,
                                   self.transformation, i)

    self.logposteriormixmean =\
      self.__predictdensitymix(self.logposteriorindivmean)

    self.posteriormixmean = np.exp(self.logposteriormixmean)
    self.posteriorindivmean = np.exp(self.logposteriorindivmean)
    self.isgridevalued = True

  def __predictdensityindiv(self, tedp, phiedp, transformed, i):
    evlogposteriorindivmean = np.empty(tedp.size)
    temp = np.empty(self.nsim)
    for j in range(tedp.size):
      temp = (phiedp[j, 0:i] * self.beta[:, 0:i, i]).sum(1) \
        - self.lognormconst[:, i]

      #get log of average of exponential of the log-likelihood for
      #each posterior simulation.
      maxtemp = temp.max()
      temp -= maxtemp
      temp = np.exp(temp)
      avgtemp = np.average(temp, weights=self.weights[:, i])
      avgtemp = np.log(avgtemp)
      avgtemp += maxtemp
      evlogposteriorindivmean[j] = avgtemp

    if transformed:
      evlogposteriorindivmean += self.laditransf(tedp)

    return evlogposteriorindivmean

  def __predictdensitymix(self, evlogposteriorindivmean):
    prefpm = np.array(evlogposteriorindivmean)
    maxprefpm = prefpm.max(axis=0)
    prefpm -= maxprefpm
    prefpm = np.exp(prefpm)
    prefpm = np.average(prefpm, axis=0, weights=self.probcomp)
    prefpm = np.log(prefpm)
    prefpm += maxprefpm
    evlogposteriormixmean = prefpm
    return evlogposteriormixmean

  def predictdensity(self, points, transformed=True, component=None,
                     logdensity=False):
    """
    Predict posterior density for estimated model.

    Parameters
    ----------
    points : scalar or 1D numpy array of points where density will be
      evaluated
    transformed : True (default) if `points` live in the sample space
      of your inputted data. `False` if your `points` live in the
      sample space of Bayesian Fourier Series sample space ([0, 1]).
    component : which individual component of the mixture to use.
      Defaults to full mixture with posterior with sieve prior
      (strongly recomended).
    logdensity : if True, will return the logdensity instead of the
      density

    Returns
    -------
    Numpy 1D array with predicted mean density
    """
    points = np.array(points, ndmin=1) #ndmin to allow call to quad
    if transformed:
      itpoints = self.itransf(points)
    else:
      itpoints = points
    if not self.transformation:
      transformed = False
    if component:
      phipoints = fourierseries(itpoints, component)
      usrlogposteriorindivmean = self.__predictdensityindiv(points,
                                                            phipoints,
                                                            transformed,
                                                            component)
      if logdensity:
        return usrlogposteriorindivmean
      else:
        return np.exp(usrlogposteriorindivmean)

    phipoints = fourierseries(itpoints, self.nmaxcomp)
    usrlogposteriorindivmean =\
      np.empty((self.nmaxcomp, points.size))
    for i in range(self.nmaxcomp):
      usrlogposteriorindivmean[i, :] =\
        self.__predictdensityindiv(points, phipoints, transformed, i)

    usrlogposteriormixmean =\
      self.__predictdensitymix(usrlogposteriorindivmean)

    if logdensity:
      return usrlogposteriormixmean
    else:
      return np.exp(usrlogposteriormixmean)

  def plot(self, ax=None, pltshow=True, component=None, **kwargs):
    """
    Plot samples.

    Parameters
    ----------
    ax : axxs to plot, defaults to axes of a new figure
    show : if True, calls matplotlib.pyplot plt.show() at end
    component : Which individual component to plot. Defaults to full
      posterior with sieve prior (mixture of individual components).
    **kwargs : aditional named arguments passed to
      matplotlib.axes.Axes.step

    Returns
    -------
    matplotlib.axes.Axes object
    """
    try:
      import matplotlib.pyplot as plt
    except ImportError:
      raise ImportError('matplotlib package required to plot')
    if not self.sfit:
      return "No samples to plot."
    if not self.isgridevalued:
      return "Must call obj.evalgrid() first."
    if not component:
      ytoplot = self.posteriormixmean
    else:
      ytoplot = self.posteriorindivmean[component, :]
    if not ax:
      ax = plt.figure().add_subplot(111)
    ax.plot(self.gridpoints, ytoplot, **kwargs)
    if pltshow:
      plt.show()
    return ax

  def __getstate__(self):
     print("You must serialize using package dill instead of pickle.")
     d = OrderedDict(self.__dict__)
     d.move_to_end("sfit")
     return d

  _smodel = None
  _smodelcode = \
  """
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
    int<lower=1> nobs; // number of data points
    matrix[nobs, nmaxcomp] phi;
    real<lower=0> hpp;
    real<lower=0> hpgamma;
  }
  transformed data {
    real x_r[0];
    int x_i[0];
    real minus_hpp_minus_half;
    real minus_hpgamma_times_i[nmaxcomp];
    minus_hpp_minus_half = -hpp - 0.5;
    for (i in 1:nmaxcomp)
      minus_hpgamma_times_i[i] = -hpgamma * i;
  }
  parameters {
    matrix[nmaxcomp, nmaxcomp] beta;
  }
  transformed parameters {
    vector[nmaxcomp] lp;
    real lognormconst[nmaxcomp];
    for (i in 1:nmaxcomp) {
      lognormconst[i] =
        log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                               to_array_1d(beta[1:i, i]),
                               x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
      lp[i] = sum(phi[, 1:i] * beta[1:i, i])
              - nobs * lognormconst[i]
              + minus_hpgamma_times_i[i];
    }
  }
  model {
    target += log_sum_exp(lp);
    for (i in 1:nmaxcomp) {
      if (i % 2 == 0)
        beta[i, ] ~ normal(0, i ^ minus_hpp_minus_half);
      else
        beta[i, ] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
    }
  }
  generated quantities {
    int<lower=1,upper=nmaxcomp> model_index;
    vector[nmaxcomp] weights;
    weights = softmax(lp);
    model_index = categorical_rng(weights);
  }
  """
