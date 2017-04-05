#!/usr/bin/env python3

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

import npcompare as npc
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

#Suppose these are your posterior obtained from MCMC
#And likelihoods are normal distribution
posterior1n = np.random.normal(0, 0.1, size=120)
posterior2n = np.random.normal(1, 0.1, size=100)

compn = npc.Compare(stats.norm.pdf, stats.norm.pdf,
                    posterior1n, posterior2n, a=-300, b=300)
compn.sample(200)
print(compn.msamples.mean())
print(len(compn))
compn.plot()
compn.sample(120)
print(compn.msamples.mean())
print(len(compn))

#Two dimensional parameter example:
#Suppose now these are your posterior obtained from MCMC
#And likelihoods are beta distribution
posterior1bAlpha = np.abs(np.random.normal(2.2, 0.1, size=80))
posterior1bBeta = np.abs(np.random.normal(2.5, 0.1, size=80))
posterior2bAlpha = np.abs(np.random.normal(1, 0.1, size=90))
posterior2bBeta = np.abs(np.random.normal(1.5, 0.1, size=90))
posterior1b = np.column_stack((posterior1bAlpha, posterior1bBeta))
posterior2b = np.column_stack((posterior2bAlpha, posterior2bBeta))

#Workaround on scipy.stats.beta.pdf so that it can accept
#the distribution parameters as a tupple/list
def f(x, params):
  return stats.beta.pdf(x, params[0], params[1])

comp12b = npc.Compare(f, f, posterior1b, posterior2b, a=0, b=1)
comp12b.sample(1000)
print(comp12b.msamples.mean())

#Another posterior very similar to posterior1b
posterior3bAlpha = np.abs(np.random.normal(2.1, 0.1, size=80))
posterior3bBeta = np.abs(np.random.normal(2.4, 0.1, size=80))
posterior3b = np.column_stack((posterior3bAlpha, posterior3bBeta))

#Compare 2-by-2
comp13b = npc.Compare(f, f, posterior1b, posterior3b, a=0, b=1)
comp13b.sample(1000)

comp23b = npc.Compare(f, f, posterior2b, posterior3b, a=0, b=1)
comp23b.sample(1000)

axx = comp12b.plot(color="blue", linewidth=2.0,
                  linestyle="-.", label="1 against 2")
comp13b.plot(axx, color="green", linewidth=2.0,
            linestyle="--", label="1 against 3")
comp23b.plot(axx, color="red", linewidth=2.0,
            linestyle=":", label="2 against 3")
plt.legend(loc='lower right', frameon=True)
