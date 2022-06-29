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

from rbnics.sampling.distributions.distribution import Distribution
from rbnics.sampling.distributions.smolyakGrid import SmolyakRule
from rbnics.sampling.distributions.tensorProductGrid import TensorProductRule
import matplotlib.pyplot as plt
import numpy as np
from rbnics.sampling.distributions import ausiliaryFunction as au

class DrawFrom(Distribution):
    def __init__(self, generator, *args, **kwargs):
        print("generator of drawfrom is ", generator)
        self.generator = generator # of a distribution in [0, 1]  ->> prende una classe del tipo di distribuzione
        self.args = args
        self.kwargs = kwargs
    
    def sample(self, box, n, typedistribution, typeGrid=0, order_flag= False):    
        print("\n\n\norder_flag is:",order_flag)
        
        if (typeGrid == 0):
            set_ = list() # of tuples
            for i in range(n):
                mu = list() # of numbers
        
                # box contiene degli intervalli tutti uguali tipo (1,3)
                for box_p in box:
                
                    # generazione della variabile aleatoria
                    mu.append(box_p[0] + self.generator(*self.args, **self.kwargs)*(box_p[1] - box_p[0]))
              
                set_.append(tuple(mu))
            
            return set_
            
        if (typeGrid == 1):

            beta_a = [self.kwargs['a'] for _ in range(len(box))] # gli dice che deve creare una lista di 9 per il primo parametro della beta
            beta_b = [self.kwargs['b'] for _ in range(len(box))]
            beta = np.concatenate(([beta_a],[beta_b]),axis=0).transpose()

            if order_flag == False:
                n_d_root = int(np.ceil(n**(1./len(box))))
                l = [n_d_root for i in range(len(box))]
                nodes, weights, m = TensorProductRule(len(box), l, typedistribution, box, beta)

            else:
                l = [n for i in range(len(box))]
                nodes, weights, m = TensorProductRule(len(box), l, typedistribution, box, beta)
            nodes = au.array2tuple(nodes,1) 
            return nodes, weights
            
        if (typeGrid == 2):
            beta_a = [self.kwargs['a'] for _ in range(len(box))] # gli dice che deve creare una lista di 9 per il primo parametro della beta
            beta_b = [self.kwargs['b'] for _ in range(len(box))]
            beta = np.concatenate(([beta_a],[beta_b]), axis=0).transpose()

            if order_flag == 0:

                q = len(box)-1
                m = 0
                while m < n:
                    q = q+1
                    nodes, weights, m = SmolyakRule(len(box), q, typedistribution, box, beta)

                #q = q -1
                nodes = au.array2tuple(nodes,1) 

            else:
                nodes, weights, m = SmolyakRule(len(box), n+1, typedistribution, box, beta)
                q = n
                nodes = au.array2tuple(nodes,1)  
            return nodes, weights
    
