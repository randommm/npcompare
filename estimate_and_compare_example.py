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

densobj1 = np.random.beta(1, 1, 40)
a = npc.EstimateBFS(densobj1)
a.sampleposterior(10000)

densobj2 = np.random.beta(0.9, 1.1, 50)
b = npc.EstimateBFS(densobj2)
b.sampleposterior(10000)

comp12 = npc.Compare.frombfs(a, b)
comp12.sampleposterior()
comp12.plot()


#Example 2
np.random.seed(10)
obs3 = np.random.normal(0.5, 3.5, 250)
obs3 = obs3[obs3 > -3]
obs3 = obs3[obs3 < 3]
densobj3 = npc.EstimateBFS(obs3, transformation={"transf": "fixed",
                                                 "vmin": -3, "vmax": 3})
densobj3.sampleposterior(10000)

obs4 = np.random.normal(-0.5, 2.5, 250)
obs4 = obs4[obs4 > -3]
obs4 = obs4[obs4 < 3]
densobj4 = npc.EstimateBFS(obs4, transformation={"transf": "fixed",
                                                 "vmin": -3, "vmax": 3})
densobj4.sampleposterior(10000)

comp34 = npc.Compare.frombfs(densobj3, densobj4,
                             transformation={"lower": -3, "upper": 3})
comp34.sampleposterior()
comp34.plot()
