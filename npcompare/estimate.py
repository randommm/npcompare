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
  Estimate univariate density using Bayesian Fourier Series.
  This method only works with data the lives in
  [0, 1], however, the class implements methods to automatically
  transform user inputted data to [0, 1]. See parameter `transform`
  below.

  Parameters
  ----------
  obs : array like
    List/tuple or numpy.array 1D array of observations. If unset,the
    constructor will not call method `fit`, and you will need to call it
    manually later to configure your model data (this can be usefull
    for using it with `scikit-learns` `GridSearchCV`).
    nmaxcomp : maximum number of components of the Fourier series
    expansion.
  nmaxcomp : integer
    Maximum number of components of the Fourier series
    expansion.
  hpp : float
    Hyperparameter p (defaults to "conservative" value of 1).
  hpgamma : float
    Hyperparameter p (defaults to "conservative" value of 0).
  transform :
    Transformation function to use. Can be either:

    String "logit" to use logit transformation. Usefull if the sample
    space is the real line. However the `obs` must actually assume
    only low values like, for example, between [-5, 5], this is due to
    the fact that inverse logit starts getting (computationally) very
    close to 1 (or 0) after 10 (or -10).

    Dictionary {"transf": "fixed", "vmin": vmin, "vmax": vmax} for fixed
    value transformation, where vmin and vmax are the minimun and
    maximum values of the sample space (for `obs`), respectivelly.

    A user defined transformation function with a dict with elements
    `transf`, `itransf` and `laditransf`, where `transf` is a function
    that transforms [0, 1] to sample space, `itransf` is its inverse,
    and `laditransf` is the log absolute derivative of `itransf`.
    These 3 functions must accept and return numpy 1D arrays.
  mixture : bool
    If True, will work with a mixture of Fourier series models
    with up to `nmaxcomponent` components (that is, a prior will be set
    on the number of components of the Fourier Series). If False, will
    work with a single Fourier series model with exactly nmaxcomponent
    components.
  **kwargs:
    Additional keyword arguments passed to method `fit`.
  """
  def __init__(self, obs=None, nmaxcomp=10, hpp=1, hpgamma=0,
               transformation=None, mixture=True, **kwargs):
    self._ismixture = mixture
    self._smodel = None

    if not "niter" in kwargs:
      kwargs["niter"] = 0
    self.__isinitialized = False
    self.fit(obs, nmaxcomp, hpp, hpgamma, transformation, **kwargs)

  def fit(self, obs, nmaxcomp=None, hpp=None, hpgamma=None,
          transformation=None, niter=5000, **kwargs):
    """
    Configure object model data and automatically calls method
      `sampleposterior` (to obtain MCMC samples from the posterior).

    Parameters
    ----------
    obs : array like
      List/tuple or numpy.array 1D array of observations.
    nmaxcomp : integer
      Maximum number of components of the Fourier series
      expansion. Defaults to the one already set in object construction.
    hpp : float
      Hyperparameter p (defaults to "conservative" value of 1).
      Defaults to the one already set in object construction.
    hpgamma : float
      Hyperparameter p (defaults to "conservative" value of 0).
      Defaults to the one already set in object construction.
    transform :
      Transformation function to use. Can be either:

      String "logit" to use logit transformation. Usefull if the sample
      space is the real line. However the `obs` must actually assume
      only low values like, for example, between [-5, 5], this is due to
      the fact that inverse logit starts getting (computationally) very
      close to 1 (or 0) after 10 (or -10).

      Dictionary {"transf": "fixed", "vmin": vmin, "vmax": vmax} for fixed
      value transformation, where vmin and vmax are the minimun and
      maximum values of the sample space (for `obs`), respectivelly.

      A user defined transformation function with a dict with elements
      `transf`, `itransf` and `laditransf`, where `transf` is a function
      that transforms [0, 1] to sample space, `itransf` is its inverse,
      and `laditransf` is the log absolute derivative of `itransf`.
      These 3 functions must accept and return numpy 1D arrays.

      Defaults to the one already set in object construction.
    niter : integer
      Number of iterations to sample. If set to 0, then will
      not call method `sample`.
    **kwargs:
      Additional keyword arguments passed to method `sample`.
    """
    #Work out transformation function
    if transformation is not None:
      if transformation == "logit":
        self.transf = scipy.special.logit #logit
        self.itransf = scipy.special.expit #invlogit
        logoftwo = np.log(2)
        self.laditransf = \
          lambda x: - np.logaddexp(logoftwo, np.logaddexp(x, -x))
      elif isinstance(transformation, dict):
        if transformation["transf"] == "fixed":
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
        raise ValueError("Unrecognized parameter transformation")

    #Save configuration parameters
    if transformation is not None or not self.__isinitialized:
      self.transformation = transformation
    if nmaxcomp is not None or not self.__isinitialized:
      self.nmaxcomp = nmaxcomp
    if hpp is not None or not self.__isinitialized:
      self.hpp = hpp
    if hpgamma is not None or not self.__isinitialized:
      self.hpgamma = hpgamma

    #Clean results
    self.sfit = None
    self.beta = None
    self.lognormconst = None
    self.weights = None
    self.nsim = None
    self.probcomp = None
    self.egresults = dict()

    #process observations
    if obs is not None:
      self.obs = np.array(obs, ndmin=1)
      self.nobs = self.obs.size
      if self.transformation is not None:
        self.itobs = self.itransf(obs)
      else:
        self.itobs = obs

      self.phi = fourierseries(self.itobs, self.nmaxcomp)
      self.modeldata = dict(nobs=self.nobs, phi=self.phi, hpp=self.hpp,
                            nmaxcomp=self.nmaxcomp, hpgamma=self.hpgamma)

      if niter != 0:
        self.sampleposterior(niter, **kwargs)
    elif self.__isinitialized:
      raise ValueError("You must supply data to `fit` method!")

    self.__isinitialized = True

    return self

  def __len__(self):
     return self.beta.shape[0]

  def compilestanmodel(self):
    """
    Compile Stan model necessary for method sample. This method is
    called automatically by obj.sample().
    """
    try:
      import pystan
    except ImportError:
      raise ImportError('pystan package required for class Estimate')
    if self.obs is None:
      raise Exception('Data is not set, must call method fit first.')
    if self._smodel is None:
      if self._ismixture:
        if EstimateBFS._smodel_mixture is None:
          EstimateBFS._smodel_mixture = pystan.StanModel(model_code = EstimateBFS._smodelcode_mixture)
        self._smodel = EstimateBFS._smodel_mixture
      else:
        if EstimateBFS._smodel_single is None:
          EstimateBFS._smodel_single = pystan.StanModel(model_code=EstimateBFS._smodelcode_single)
        self._smodel = EstimateBFS._smodel_single

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
      Number of CPUs to be used in parallel. If -1 (default), all CPUs
      will be used.
    tolrhat :
      Maximum tolerable distance an Rhat of any sampled parameter can
      have from 1 (we will resample model approximatedly 10% more
      iterations until this convergence criteria is met).
    **kwargs :
      Additional keyword arguments passed to pystan (e.g.: refresh
      parameter to configure sampler printing status).

    Returns
    -------
    None
    """
    self.compilestanmodel()

    while True:
      self.sfit = self._smodel.sampling(data=self.modeldata,
                                        n_jobs=njobs, chains=nchains,
                                        iter=niter, **kwargs)
      irhat = self.sfit.summary()['summary_colnames'].index("Rhat")
      drhat1 = max(abs(self.sfit.summary()["summary"][:, irhat] - 1))
      if drhat1 < tolrhat:
        break
      niter += niter // 10
      print("Model failed to converge given your tolrhat of ", tolrhat,
            "; the observed maximum distance of an Rhat from 1 was ",
            drhat1, "; retrying sampling with ", niter, "iterations")

    self.__processfit()
    self.egresults = dict()

  def __processfit(self):
    if not self.sfit:
      raise Exception("This function cannot be called before obj.sampleposterior()")

    #Code commented due to some weird bug on pystan.extract
    if not self._ismixture:
      self.beta = self.sfit.extract("beta")["beta"]
      self.lognormconst = \
      self.sfit.extract("lognormconst")["lognormconst"]
      self.nsim = self.beta.shape[0]
      return
    #self.weights = self.sfit.extract("weights")["weights"]
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

  def evalgrid(self, gridsize=1000):
    """
    Calculates posterior estimated mean density value at grid points so
    they can be used later by method plot or directly by the user.

    Parameters
    ----------
    gridsize : size of the grid.

    Returns
    -------
    None

    Notes
    -----
    The results are store in an instance variable called
    `egresults` which is a dictionary with the following objects:

      If object was initialized with mixture=True, you will have:

        Log densities for full mixture of components stored as
        `logdensitymixmean`.

        Densities for full mixture of components stored as
        `densitymixmean`.

        Log densities for individual components stored as
        `logdensityindivmean`.

        Densities for individual components stored as `densityindivmean`.

      If object was initialized with mixture=False, you will have:

        Log densities stored as `logdensitymean`.

        Densities stored as `densitymean`.
    """
    transformed = self.transformation is not None
    if self.beta is None:
      raise Exception("No MCMC samples available, you must call"
                      "obj.sampleposterior() first.")

    self.egresults = dict()
    gddict = self.egresults
    #Construct density evaluation points for method evalgrid
    gddict["gridpoints_internal_bfs"] = np.linspace(0, 1, gridsize)
    gridpointsibfs = gddict["gridpoints_internal_bfs"]
    gridpointsibfs[0] += 1.0 / 1e10 / gridsize
    gridpointsibfs[-1] -= 1.0 / 1e10 / gridsize
    phidp = fourierseries(gridpointsibfs, self.nmaxcomp)
    if transformed:
      gddict["gridpoints"] = self.transf(gridpointsibfs)
    else:
      gddict["gridpoints"] = gridpointsibfs
    gridpoints = gddict["gridpoints"]

    #Special case of mixture=False
    if not self._ismixture:
      gddict["logdensitymean"] = \
        self.__predictdensitysingle(gridpoints, phidp,
                                    transformed)
      gddict["densitymean"] = np.exp(gddict["logdensitymean"])
      return

    #Empty var to store results
    gddict["logdensityindivmean"] =\
      np.empty((self.nmaxcomp, gridsize))

    #Eval individual components
    for i in range(self.nmaxcomp):
      gddict["logdensityindivmean"][i, :] =\
        self.__predictdensityindiv(gridpoints, phidp,
                                   transformed, i)

    #Eval mix density
    gddict["logdensitymixmean"] =\
      self.__predictdensitymix(gddict["logdensityindivmean"])

    gddict["densitymixmean"] = np.exp(gddict["logdensitymixmean"])
    gddict["densityindivmean"] = np.exp(gddict["logdensityindivmean"])

  def __predictdensityindiv(self, tedp, phiedp, transformed, i):
    evlogdensityindivmean = np.empty(tedp.size)
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
      evlogdensityindivmean[j] = avgtemp

    if transformed:
      evlogdensityindivmean += self.laditransf(tedp)

    return evlogdensityindivmean

  def __predictdensitysingle(self, tedp, phiedp, transformed):
    evlogdensitymean = np.empty(tedp.size)
    temp = np.empty(self.nsim)
    for j in range(tedp.size):
      temp = (phiedp[j, :] * self.beta).sum(1) - self.lognormconst

      #get log of average of exponential of the log-likelihood for
      #each posterior simulation.
      maxtemp = temp.max()
      temp -= maxtemp
      temp = np.exp(temp)
      avgtemp = np.average(temp)
      avgtemp = np.log(avgtemp)
      avgtemp += maxtemp
      evlogdensitymean[j] = avgtemp

    if transformed:
      evlogdensitymean += self.laditransf(tedp)

    return evlogdensitymean

  def __predictdensitymix(self, evlogdensityindivmean):
    prefpm = np.array(evlogdensityindivmean)
    maxprefpm = prefpm.max(axis=0)
    prefpm -= maxprefpm
    prefpm = np.exp(prefpm)
    prefpm = np.average(prefpm, axis=0, weights=self.probcomp)
    prefpm = np.log(prefpm)
    prefpm += maxprefpm
    evlogdensitymixmean = prefpm
    return evlogdensitymixmean

  def score(self, points, transformed=True, component=None):
    """
    Return sum of average logdensity of points.

    This is equivalent to calling:
    ::

      `obj.evaluate(points, logdensity=True, transformed,
      component).sum()[()]`

    Parameters
    ----------
    points : array like or scalar
      Scalar or 1D numpy array of points where density will be
      evaluated.
    transformed : bool
      True (default) if `points` live in the sample space
      of your inputted data. `False` if your `points` live in the
      sample space of Bayesian Fourier Series sample space ([0, 1]).
    component : NoneType or integer
      Which individual component of the mixture to use.
      Set to None (default) to full mixture with posterior with sieve
      prior (strongly recomended).
      Ignored if object was initialized with mixture=False.

    Returns
    -------
    Numpy float with sum of predicted mean log densities.
    """
    return self.evaluate(points, True, transformed, component).sum()[()]

  def evaluate(self, points, logdensity=False, transformed=True,
               component=None):
    """
    Predict posterior density for estimated model.

    Parameters
    ----------
    points : array like or scalar
      Scalar or 1D numpy array of points where density will be
      evaluated.
    transformed : bool
      True (default) if `points` live in the sample space
      of your inputted data. `False` if your `points` live in the
      sample space of Bayesian Fourier Series sample space ([0, 1]).
    component : NoneType or integer
      Which individual component of the mixture to use.
      Set to None (default) to full mixture with posterior with sieve
      prior (strongly recomended).
      Ignored if object was initialized with mixture=False.
    logdensity : bool
      If True, will return the logdensity instead of the density.

    Returns
    -------
    Numpy 1D array with predicted mean density.
    """
    if self.beta is None:
      raise Exception("No MCMC samples available, you must call"
                      "obj.sampleposterior() first.")
    points = np.array(points, ndmin=1) #ndmin to allow call to quad
    if self.transformation is None:
      transformed = False
    if transformed:
      itpoints = self.itransf(points)
    else:
      itpoints = points

    #Special case of mixture=False
    if not self._ismixture:
      phipoints = fourierseries(itpoints, self.nmaxcomp)
      usrlogdensitymean = \
        self.__predictdensitysingle(points, phipoints, transformed)
      if logdensity:
        return usrlogdensitymean
      else:
        return np.exp(usrlogdensitymean)

    #Special case for single component of mixture
    if component:
      phipoints = fourierseries(itpoints, component)
      usrlogdensityindivmean = self.__predictdensityindiv(points,
                                                          phipoints,
                                                          transformed,
                                                          component)
      if logdensity:
        return usrlogdensityindivmean
      else:
        return np.exp(usrlogdensityindivmean)

    phipoints = fourierseries(itpoints, self.nmaxcomp)
    usrlogdensityindivmean =\
      np.empty((self.nmaxcomp, points.size))
    for i in range(self.nmaxcomp):
      usrlogdensityindivmean[i, :] =\
        self.__predictdensityindiv(points, phipoints, transformed, i)

    usrlogdensitymixmean =\
      self.__predictdensitymix(usrlogdensityindivmean)

    if logdensity:
      return usrlogdensitymixmean
    else:
      return np.exp(usrlogdensitymixmean)

  def plot(self, ax=None, pltshow=True, component=None, **kwargs):
    """
    Plot samples.

    Parameters
    ----------
    ax : matplotlib axes
      Axis to plot, defaults to axes of a new figure.
    show : bool
      If True, calls matplotlib.pyplot plt.show() at end.
    component : NoneType or integer
      Which individual component of the mixture to use.
      Set to None (default) to full mixture with posterior with sieve
      prior (strongly recomended).
      Ignored if object was initialized with mixture=False.
    **kwargs :
      Aditional named arguments passed to matplotlib.axes.Axes.step.

    Returns
    -------
    matplotlib.axes.Axes object
    """
    try:
      import matplotlib.pyplot as plt
    except ImportError:
      raise ImportError('matplotlib package required to plot')
    if not len(self.egresults):
      return "Must call obj.evalgrid() first."
    if not self._ismixture:
      ytoplot = self.egresults["densitymean"]
    elif component is None:
      ytoplot = self.egresults["densitymixmean"]
    else:
      ytoplot = self.egresults["densityindivmean"][component, :]
    if ax is None:
      ax = plt.figure().add_subplot(111)
    ax.plot(self.egresults["gridpoints"], ytoplot, **kwargs)
    if pltshow:
      plt.show()
    return ax

  def __getstate__(self):
     print("You must serialize using package dill instead of pickle.")
     d = OrderedDict(self.__dict__)
     d.move_to_end("sfit")
     return d

  def __setstate__(self, d):
     if "_ismixture" not in d.keys():
       d["_ismixture"] = True
     if "isgridevalued" in d.keys():
       del(d["isgridevalued"])
     if "egresults" not in d.keys():
       d["egresults"] = dict()
     for objname in ["densityindivmean", "logdensityindivmean",
                     "densitymixmean", "logdensitymixmean",
                     "densitymean", "logdensitymean",
                     "gridpoints_internal_bfs", "gridpoints"]:
       if objname in d.keys():
         d["egresults"][objname] = d[objname]
         del(d[objname])
     self.__dict__ = d

  _smodel_mixture = None
  _smodel_single = None

  _smodelcode_mixture = \
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

  _smodelcode_single = \
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
  }
  transformed data {
    real x_r[0];
    int x_i[0];
    real minus_hpp_minus_half;
    minus_hpp_minus_half = -hpp - 0.5;
  }
  parameters {
    vector[nmaxcomp] beta;
  }
  transformed parameters {
    real lognormconst;
    lognormconst =
      log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                             to_array_1d(beta),
                             x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
  }
  model {
    target += sum(phi * beta) - nobs * lognormconst;
    for (i in 1:nmaxcomp) {
      if (i % 2 == 0)
        beta[i] ~ normal(0, i ^ minus_hpp_minus_half);
      else
        beta[i] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
    }
  }
  """
