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

from ._globals import *
from npcompare.fourierseries import fourierseries

class EstimateBFS:
  """
  Attention: this class requires a patched version of pystan.

  Estimate univariate density using Bayesian Fourier Series
  with a sieve prior

  Parameters
  ----------
  obs : list/tuple or numpy.array 1D array of observations.
  nmaxcomp : maximum number of components of the Fourier series
    expansion.
  hpp : hiperparameter p (defaults to "conservative" value of 1).
  hpgamma : hiperparameter p (defaults to "conservative" value of 0).
  """
  smodel = None
  dp = np.arange(1001)/1000

  def __init__(self, obs=None, nmaxcomp=10, hpp=1, hpgamma=0):
    obs = np.array(obs)
    self.phi = fourierseries(obs, nmaxcomp)
    self.phidp = fourierseries(self.dp, nmaxcomp)
    self.nmaxcomp = nmaxcomp
    self.obs = np.array(obs)
    self.nobs = len(self.obs)
    self.hpp = hpp
    self.hpgamma = hpgamma
    self.modeldata = dict(nobs=self.nobs, phi=self.phi, hpp=hpp,
                          nmaxcomp=nmaxcomp, hpgamma=hpgamma)
    self.probcomp = None
    self.sfit = None

  def __len__(self):
     return self.samples.size

  def sample(self, niter=1000, nchains=4, njobs=-1, **kwargs):
    """
    Samples from posterior.

    Parameters
    ----------
    niter : number of simulations to be draw.
    nchains : number of MCMC chains.
    njobs : number of CPUs to be used in parallel. If -1 (default),
      all CPUs will be used.
    **kwargs : aditional named arguments passed to pystan (e.g.:
      refresh parameter to configure sampler printing status).

    Returns
    -------
    None
    """
    print('Attention: this class requires a patched version of pystan.')
    try:
      import pystan
    except ImportError:
      raise ImportError('pystan package required for class Estimate')

    if not self.smodel:
      EstimateBFS.smodel = \
        pystan.StanModel(model_code=self.__modelcode)

    self.smodel = EstimateBFS.smodel
    self.sfit = self.smodel.sampling(data=self.modeldata, n_jobs=njobs,
                                     chains=nchains, iter=niter,
                                     **kwargs)

    self.__processfit()

  def __processfit(self):
    if not self.sfit:
      Exception("This function cannot be called before obj.sample")

    self.beta = self.sfit.extract("beta")["beta"]
    self.nsim = self.beta.shape[0]
    self.log_norm_const = \
      self.sfit.extract("log_norm_const")["log_norm_const"]
    self.weights = self.sfit.extract("weights")["weights"]
    self.probcomp = self.sfit.extract("weights")["weights"].mean(0)
    self.logposteriorindivmean =\
      np.empty((self.nmaxcomp, len(self.phidp)))

    temp = np.empty(self.nsim)
    for i in range(self.nmaxcomp):
      for j in range(len(self.dp)):
        for k in range(self.nsim):
          temp[k] = sum(self.phidp[j, 0:i] * self.beta[k, 0:i, i]) \
            - self.log_norm_const[k, i]

        #get log of average of exponential of the log-likelihood for
        #each posterior simulation.
        maxtemp = temp.max()
        temp -= maxtemp
        temp = np.exp(temp)
        avgtemp = np.average(temp, weights=self.weights[:, i])
        avgtemp = np.log(avgtemp)
        avgtemp += maxtemp
        self.logposteriorindivmean[i, j] = avgtemp

    prefpm = np.array(self.logposteriorindivmean)
    maxprefpm = prefpm.max(axis=0)
    prefpm -= maxprefpm
    prefpm = np.exp(prefpm)
    prefpm = np.average(prefpm, axis=0, weights=self.probcomp)
    prefpm = np.log(prefpm)
    prefpm += maxprefpm
    self.logposteriormixmean = prefpm

    self.posteriormixmean = np.exp(self.logposteriormixmean)
    self.posteriorindivmean = np.exp(self.logposteriorindivmean)

  def plot(self, ax=None, pltshow=True, component=None,
           *args, **kwargs):
    """
    Plot samples.

    Parameters
    ----------
    ax : axxs to plot, defaults to axes of a new figure
    show : if True, calls matplotlib.pyplot plt.show() at end
    component : Which individual component to plot. Defaults to full
      posterior with sieve prior (mixture of individual components).
    *args : aditional arguments passed to matplotlib.axes.Axes.step
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
    if not component:
      ytoplot = self.posteriormixmean
    else:
      ytoplot = self.posteriorindivmean[component, :]
    if not ax:
      ax = plt.figure().add_subplot(111)
    ax.plot(self.dp, ytoplot, **kwargs)
    if pltshow:
      plt.show()
    return ax

  __modelcode = \
  """
  functions {
    real lkk(real x, vector beta) {
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
    real glkk(real x, vector beta, int n) {
      real result_;
      result_ = 0;
      for (i in 1:num_elements(beta)) {
        if (i % 2 == 0)
          result_ = result_ + beta[i] * cos(i * pi() * x);
        else
          result_ = result_ + beta[i] * sin((i + 1) * pi() * x);
      }

      if (n % 2 == 0)
        result_ = exp(result_ * sqrt2()) *
                   sqrt2() * cos(n * pi() * x);
      else
        result_ = exp(result_ * sqrt2()) *
                   sqrt2() * sin((n + 1) * pi() * x);

      return result_;
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
    real log_norm_const[nmaxcomp];
    for (i in 1:nmaxcomp) {
      log_norm_const[i] =
        log(integrate_1d_grad(lkk, glkk, 0, 1, beta[1:i, i]));
      lp[i] = sum(phi[, 1:i] * beta[1:i, i])
              - nobs * log_norm_const[i]
              + minus_hpgamma_times_i[i];
    }
  }
  model {
    target += lp;
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
