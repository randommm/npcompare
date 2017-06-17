#----------------------------------------------------------------------
# Copyright 2017 Marco Inacio <npcompare@marcoinacio.com>
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
import npcompare as npc
import scipy.stats as stats

np.random.seed(12)

#Example 1
obs0 = np.random.normal(0.1, 1, size=60)
obs1 = np.random.normal(0.05, 1, size=50)
test1 = npc.EstimateLindleyBFS(obs0, obs1, nmaxcomp=7,
                               #46% a priori probability that each
                               #dataset came from different populations
                               hplindley=.46,
                               #Just like with EstimateBFS, you can work
                               #with a mixture of Bayesian Fourier
                               #series, or with a single one
                               mixture=True,
                               #You can also work with transformation of
                               #data as usual, see the example file for
                               #EstimateBFS for details.
                               transformation="logit")
test1.compilestanmodel()
test1.sampleposterior(niter=10000, nchains=2, refresh=100, init_r=0.2)

#Print a posteriori probability, where the first element of the array
#is the probability that each dataset came from different populations
#and the second element of the array is the probability that both
#datasets came from same population
print(test1.problindley)

#You can also get the indivual EstimateBFS objects and work with them
#as usual where
#test1.bfs0: EstimateBFS object fitted to dataset 1
#test1.bfs1: EstimateBFS object fitted to dataset 2
#test1.bfsconcat: EstimateBFS object fitted both datasets
#concatenated (that is, assuming a priori that both datasets came from
#the same population).
test1.bfs0.evalgrid()
test1.bfs1.evalgrid()
test1.bfsconcat.evalgrid()
p = test1.bfs0.plot(color="red")
test1.bfs1.plot(p, color="green")
test1.bfsconcat.plot(p, color="yellow")
p.set_xlim(-7,7)
