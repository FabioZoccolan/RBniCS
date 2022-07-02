# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from math import ceil #ceil round the number up to the next integer
from numpy import linspace #return evenly spaced number over a specified interval
import itertools # it implements a number of iterator building blocks, it standardizes a core set of fast, memory efficient tools that are useful by themselves or in combination. 
from rbnics.sampling.distributions.distribution import Distribution

class EquispacedDistribution(Distribution):
    def sample(self, box, n, typeGrid, order_flag):
        print("I am in sample of EquispacedDistribution\n")
        n_P_root = int(ceil(n**(1./len(box))))
        grid = list() # of linspaces
        for box_p in box:
            grid.append(linspace(box_p[0], box_p[1], num=n_P_root).tolist()) #Returns num evenly spaced samples, calculated over the interval [start, stop]. tolist(), used to convert the data elements of an array into a list. 
        set_itertools = itertools.product(*grid) #cartesian product, equivalent to a nested for-loop
        set_ = list() # of tuples
        for mu in set_itertools:
            set_.append(mu)
        return set_
