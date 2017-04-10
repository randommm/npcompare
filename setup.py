#!/usr/bin/env python3

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

from distutils.core import setup

setup(name='npcompare',
      version='0.6.0',
      description='Nonparametric methods for density estimation and comparison',
      author='Marco Inacio',
      author_email='pythonpackages@marcoinacio.com',
      url='http://npcompare.marcoinacio.com/',
      packages=['npcompare'],
      keywords = ['nonparametric', 'density comparision', 'density estimation'],
      license='GPL3',
      requires=['numpy']
     )
