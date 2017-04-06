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

class Estimate:
  """
  Not implemented yet.

  Estimate univariate density using Bayesian Fourier Series
  with a sieve prior

  Parameters
  ----------
  data : observations
  nsim : number of simulations to draw
  """
  def __init__(self, data, nsim):
    raise Exception('Class Estimate is not implemented yet')
    try:
      import pystan
    except ImportError:
      raise ImportError('pystan package required for class Estimate')
    self.pystan = pystan

  def __len__(self):
     return self.samples.size

  def sample(self, nsim=1000, printstatus=100):
    """
    Samples from posterior.

    Parameters
    ----------
    nsim : number of simulations to be draw.
    printstatus : interval of samples to print the amount of
      samples obtained so far.
      Set to 0 to disable printing.

    Returns
    -------
    None
    """
    result = np.empty(nsim)

    self.msamples = np.hstack([self.msamples, result])

  def plot(self, ax=None, show=True, *args, **kwargs):
    """
    Plot samples.

    Parameters
    ----------
    ax : axxs to plot, defaults to axes of a new figure
    show : if True, calls matplotlib.pyplot plt.show() at end
    *args : aditional arguments passed to matplotlib.axes.Axes.step
    **kwargs : aditional named arguments passed to
      matplotlib.axes.Axes.step

    Returns
    -------
    matplotlib.axes.Axes object
    """
    try:
      import matplotlib.pyplot
    except ImportError:
      raise ImportError('matplotlib package required to plot')
