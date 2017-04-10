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

class Compare:
  """
  Compare two samples.

  Parameters
  ----------
  f1 : likelihood function for the first population.
  f2 : likelihood function for the second population.
  psamples1 : must be either a one dimensional numpy.array
    (in which case, each element will be passed one at a time to f1)
    or numpy matrix or two dimensional numpy.array (in which case,
    each row will be be passed one at a time to f1).
  psamples2 : analog to psamples1.
  weights1 : Give weights to posterior psamples1.
    Set to None if each posterior sample has the same weight
    (the usual case for MCMC methods).
  weights2 : analog to weights2.
  metric : The metric function to be used
    Defaults to:

    def metric(f1, f2, param1, param2): return quad(lambda x: (f1(x, param1) - f2(x, param2))**2, a, b)[0]

    Can be set to a user-defined function of the same signature.
  a : lower integration limit passed to default metric function.
  b : upper integration limit passed to default metric function.
  """
  def __init__(self, f1, f2, psamples1, psamples2, a=0, b=1,
               weights1=None, weights2=None, metric=None):
    self.f1 = f1
    self.f2 = f2
    self.psamples1 = np.array(psamples1)
    self.psamples2 = np.array(psamples2)
    self.weights1 = np.array(weights1)
    self.weights2 = np.array(weights2)
    if metric == None:
      try:
        from scipy.integrate import quad
      except ImportError:
        raise ImportError('scipy package required with default metric')
      def metric (f1, f2, param1, param2):
        return quad(lambda x: (f1(x, param1) - f2(x, param2))**2,
                               a, b)[0]

    self.metric = metric

    self.msamples = np.array([], dtype=np.float64)

  def __len__(self):
     return self.msamples.size

  def sample(self, niter=1000, refresh=100):
    """
    Compare two samples.

    Parameters
    ----------
    niter : number of simulations to be draw.
    refresh : interval of samples to print the amount of
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
    ax : axxs to plot, defaults to axes of a new figure
    show : if True, calls matplotlib.pyplot plt.show() at end
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
    if len(self.msamples) == 0:
      return "No metric samples to plot"
    smsamples = np.sort(self.msamples)
    if not ax:
      ax = plt.figure().add_subplot(111)
    ax.step(smsamples, np.arange(len(self)) / len(self), **kwargs)
    if pltshow:
      plt.show()
    return ax

