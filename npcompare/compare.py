#----------------------------------------------------------------------
# Copyright 2017 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
from npcompare.fourierseries import fourierseries
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
    or a two dimensional numpy.array (in which case, each row will be
    passed one at a time to f1).
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
      def metric (f1, f2, param1, param2):
        return quad(lambda x: (f1(x, param1) - f2(x, param2))**2,
                               lower, upper)[0]

    self.metric = metric

    self.msamples = np.array([], dtype=np.float64)

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
      If set to None, metric will be evaluated in [0, 1] without
      transformation (EstimateBFS class documentation).
      Otherwise, transformation will be applied.

      The parameter can also be set to a dictionary with the lower and
      upper boundaries of integration manually set, that is:
      {"lower": lower, "upper": upper} (the sample space of observed
      data).

      Ignored if bfsobj1 has no transformation.
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

    for (bfsobj, psamples, weights) in \
      ((bfsobj1, psamples1, weights1), (bfsobj2, psamples2, weights2)):
      for i in range(bfsobj.beta.shape[0]):
        for k in range(bfsobj.nmaxcomp):
          wl = i * bfsobj.nmaxcomp + k
          psamples[wl, 0] = bfsobj.lognormconst[i, k]
          psamples[wl, 1:(k+2)] = bfsobj.beta[i, 0:(k+1), k]
          psamples[wl, (k+2):] = 0.0
          weights[wl] = bfsobj.weights[i, k]

    if transformation is not None and bfsobj1.laditransf is not None:
      def f1(x, psample):
        logd = (fourierseries(x, bfsobj1.nmaxcomp) * \
          psample[1:]).sum() - psample[0] + \
          bfsobj1.laditransf(bfsobj1.itransf(x))
        return np.exp(logd)
      def f2(x, psample):
        logd = (fourierseries(x, bfsobj2.nmaxcomp) * \
          psample[1:]).sum() - psample[0] + \
          bfsobj2.laditransf(bfsobj2.itransf(x))
        return np.exp(logd)
      if isinstance(transformation, dict):
        lower = transformation["lower"]
        upper = transformation["upper"]
      else:
        lower = bfsobj1.transf(0)
        upper = bfsobj1.transf(1)
    else:
      def f1(x, psample):
        logd = (fourierseries(x, bfsobj1.nmaxcomp) * \
          psample[1:]).sum() - psample[0]
        return np.exp(logd)
      def f2(x, psample):
        logd = (fourierseries(x, bfsobj2.nmaxcomp) * \
          psample[1:]).sum() - psample[0]
        return np.exp(logd)
      lower = 0
      upper = 1

    return cls(f1, f2, psamples1, psamples2, lower, upper,
               weights1, weights2, metric)

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
    None
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

  def plot(self, ax=None, pltshow=True, **kwargs):
    """
    Plot empirical CDF of metric samples.

    Parameters
    ----------
    ax : matplotlib axes
      Axis to plot, defaults to axes of a new figure.
    show : bool
      If True, calls matplotlib.pyplot plt.show() at end.
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
    if self.msamples.size == 0:
      return "No metric samples to plot"
    smsamples = np.sort(self.msamples)
    if ax is None:
      ax = plt.figure().add_subplot(111)
    ax.step(smsamples,
            np.arange(self.msamples.size) / self.msamples.size,
            **kwargs)
    if pltshow:
      plt.show()
    return ax

