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


def fourierseries(x, ncomponents):
  """Calculate Fourier Series Expansion.

  Parameters
  ----------
  x : 1D numpy.array, list or tuple of numbers to calculate
    fourier series expansion
  ncomponents : number of components of the series

  Returns
  ----------
  2D numpy.array where each line is the Fourier series expansion of each
   component of x.
  """
  from numpy import sqrt, sin, cos, pi
  x = np.array(x, ndmin=1)
  results = np.array(np.empty((x.size, ncomponents)))

  for i in range(x.size):
    for j in range(ncomponents):
      if j%2 == 0:
        results[i, j] = sqrt(2) * sin((j+2) * pi * x[i])
      else:
        results[i, j] = sqrt(2) * cos((j+1) * pi * x[i])

  return(results)
