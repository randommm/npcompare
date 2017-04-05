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
from numpy import sqrt, sin, cos, pi

def calculate_basis(z, max_):
  x = np.array(z, copy=True)
  results = np.matrix(np.empty((len(x), max_)), copy=True)

  for i in range(len(x)):
    for j in range(max_):
      if j%2 == 0:
        results[i, j] = sqrt(2) * sin((j+2) * pi * x[i])
      else:
        results[i, j] = sqrt(2) * cos((j+1) * pi * x[i])

  return(results)
