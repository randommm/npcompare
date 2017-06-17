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
obs = np.random.beta(1, 1, 100)
densobj1 = npc.EstimateBFS(obs, 5)
densobj1.sampleposterior(10000)

#evalualte estimated at a grid of points (needed to plot object)
densobj1.evalgrid()

#Plot estimated density:
p = densobj1.plot(color="red")
#Plot individual component of the mixture:
densobj1.plot(p, 4, color="green")
densobj1.plot(p, 3, color="blue")
#Plot true density:
p.plot(densobj1.egresults["gridpoints"],
       stats.beta.pdf(densobj1.egresults["gridpoints"], 1, 1),
       color="yellow")


#Example 2
obs=np.random.normal(0, 1, 90)
densobj2 = npc.EstimateBFS(obs, 5, transformation="logit")
densobj2.sampleposterior(10000)

densobj2.evalgrid()
p = densobj2.plot()
densobj2.plot(p, 4)
densobj2.plot(p, 3)
p.plot(densobj2.egresults["gridpoints"],
       stats.norm.pdf(densobj2.egresults["gridpoints"], 0, 1))


#Example 3
np.random.seed(10)
obs = np.random.normal(0.5, 3.5, 120)
obs = obs[obs > -3]
obs = obs[obs < 3]
densobj3 = npc.EstimateBFS(obs, transformation={"transf": "fixed",
                                                "vmin": -3, "vmax": 3})
densobj3.sampleposterior(10000)
densobj3.evalgrid()
p = densobj3.plot()
densobj3.plot(p, 4)
densobj3.plot(p, 3)
p.plot(densobj3.egresults["gridpoints"],
       (stats.norm.pdf(densobj3.egresults["gridpoints"], 0.5, 3.5) /
       (stats.norm.cdf(3, 0.5, 3.5) - stats.norm.cdf(-3, 0.5, 3.5))))


from scipy.integrate import quad
#Check if integrates to 1
#You could call:
#quad(lambda x: densobj3.evalposterior(x), -3, 3)
#But the recommended faster way is:
quad(lambda x: densobj3.evaluate(x, transformed=False), 0, 1)

#Estimate mean and variance
#Here you must use transformed space
estmean = quad(lambda x: x * densobj3.evaluate(x), -3, 3)[0]
estvar = quad(lambda x: x ** 2.0 * densobj3.evaluate(x),
              -3, 3)[0] - estmean ** 2.0


#Example 4
#Set mixture=False to work with a single Bayesian Fourier Series
obs=np.random.normal(-0.4, 1.2, 115)
densobj4 = npc.EstimateBFS(obs, mixture=False, nmaxcomp=10,
                           transformation="logit")
densobj4.sampleposterior(10000)
densobj4.evalgrid()
p = densobj4.plot()
p.plot(densobj4.egresults["gridpoints"],
       stats.norm.pdf(densobj4.egresults["gridpoints"], -0.4, 1.2))
