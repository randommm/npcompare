#----------------------------------------------------------------------
# Copyright 2017 Marco Inacio <npcompare@marcoinacio.com>
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
from scipy.integrate import quad

def compare(f1, f2, samples1, samples2, n_sim=1000,
            weights1=None, weights2=None, metric=None,
            a=0, b=1, printstatus=100):
  """Compare two samples.

  Parameters
  ----------
  f1: log-likelihood for the first population
  f2: log-likelihood for the second population

  samples1: must be either a one dimensional numpy.array (in which case, each element will be passed one at a time to f1) or numpy matrix or two dimensional numpy.array (in which case, each row will be be passed one at a time to f1).
  samples2: analog to samples1.

  weights1: Give weights to posterior samples1. Set to None if each posterior sample has the same weight (the usual case for MCMC methods).
  weights2: analog to weights2.

  n_sim: number of simulations to be draw.

  metric:
    The metric function to be used
    Defaults to:
    def metric(f1, f2, param1, param2):
      return quad(lambda x: (f1(x, param1) - f2(x, param2))**2, a, b)[0]
    Can be set to a user-defined function of the same signature

  a: lower integration limit passed to default metric function
  b: upper integration limit passed to default metric function
  printstatus: interval of samples to print the amount of samples obtained so far.
    Set to 0 to disable printing.

  Returns
  -------
  numpy.array of samples of the metric."""
  if metric == None:
    def metric(f1, f2, param1, param2):
      return quad(lambda x: (f1(x, param1) - f2(x, param2))**2, a, b)[0]

  result = np.empty(n_sim)
  samples1 = np.array(samples1, copy=True)
  samples2 = np.array(samples2, copy=True)
  samples1_index = np.random.choice(np.arange(samples1.shape[0]), n_sim, p=weights1)
  samples2_index = np.random.choice(np.arange(samples2.shape[0]), n_sim, p=weights2)

  samples1 = samples1[samples1_index]
  samples2 = samples2[samples2_index]
  for i in range(n_sim):
    result[i] = metric(f1, f2, samples1[i], samples2[i])
    if (printstatus):
      if (not i%printstatus):
        print(i, "samples generated")

  return result
