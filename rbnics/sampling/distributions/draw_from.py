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
        print("In __init__ of DrawFrom")
        print("generator of drawfrom is ", generator)
        self.generator = generator # of a distribution in [0, 1]  ->> prende una classe del tipo di distribuzione
        self.args = args
        self.kwargs = kwargs
        print("kwargs=", self.kwargs)
        print("so the keys are", kwargs.keys)
        for x in kwargs.keys():
            print(x)

    def sample(self, box, n, typedistribution, typeGrid=0, order_flag=False, is_loguniform=False):     #typeGrid=0
        print("in drawfrom.sample")
        print("\n\n\norder_flag is:", order_flag)
        #print("is_loguniform=", self.kwargs["is_loguniform"])
        print("box=", box)
        if (typeGrid == 0 and is_loguniform == False):
            # print("In typeGrid==0: Montecarlo") #added
            set_ = list() # of tuples
            for i in range(n):
                mu = list() # of numbers
        
                # box contiene degli intervalli tutti uguali tipo (1,3)
                for box_p in box:
                
                    # generazione della variabile aleatoria
                    mu.append(box_p[0] + self.generator(*self.args, **self.kwargs)*(box_p[1] - box_p[0]))
              
                set_.append(tuple(mu))
            
            return set_

        if (typeGrid == 0 and is_loguniform == True):
            print("In typeGrid==0: Montecarlo") #added
            print("is_loguniform=", is_loguniform)
            set_ = list() # of tuples
            for i in range(n):
                mu = list() # of numbers
        
                # box contiene degli intervalli tutti uguali tipo (1,3)
                for box_p in box:
                    
                    # generazione della variabile aleatoria
                    scale = -(np.log(box_p[1])-np.log(box_p[0]))/np.log(1e-4)
                    mu.append(box_p[0]*(self.generator(*self.args, **self.kwargs)/(1e-4))**scale)
              
                set_.append(tuple(mu))
            
            return set_
            
        if (typeGrid == 1):
            print("In typegrid==1 of DrawFrom")
            print("range(len(box))=", range(len(box)))
            print("the keys are", self.kwargs.keys())
            #beta_a = [self.kwargs['a'] for _ in range(len(box))] # gli dice che deve creare una lista di 9 per il primo parametro della beta
            #beta_b = [self.kwargs['b'] for _ in range(len(box))]
            #beta = np.concatenate(([beta_a],[beta_b]),axis=0).transpose()

            param={}
            for key in self.kwargs.keys():
                param[key]= [self.kwargs[key] for _ in range(len(box))] # gli dice che deve creare una lista di 9 per il primo parametro della beta     
            print('param=', param)
            for key in param.keys():
                print("param[", key, "]=", param[key])       
            help = ()
            for key in param:
                help=help+([param[key]],)
            beta2 = np.concatenate(help, axis=0).transpose()

            if order_flag == False:
                n_d_root = int(np.ceil(n**(1./len(box))))
                l = [n_d_root for i in range(len(box))]
                nodes, weights, m = TensorProductRule(len(box), l, typedistribution, box, beta2, is_loguniform)

            else:
                l = [n for i in range(len(box))]
                nodes, weights, m = TensorProductRule(len(box), l, typedistribution, box, beta2, is_loguniform)
            nodes = au.array2tuple(nodes,1) 
            return nodes, weights
            
        if (typeGrid == 2):
            
            print("In typegrid==2 of DrawFrom")
            
            #beta_a = [self.kwargs['a'] for _ in range(len(box))] # gli dice che deve creare una lista di 9 per il primo parametro della beta
            #beta_b = [self.kwargs['b'] for _ in range(len(box))]
            #beta = np.concatenate(([beta_a],[beta_b]), axis=0).transpose()
            #print("beta_a, beta_b=", beta_a, beta_b)
            #print("([beta_a],[beta_b])=", ([beta_a],[beta_b]))

            param={}
            for key in self.kwargs.keys():
                param[key]= [self.kwargs[key] for _ in range(len(box))] # gli dice che deve creare una lista di 9 per il primo parametro della beta     
            print('param=', param)
            for key in param.keys():
                print("param[", key, "]=", param[key])       
            help = ()
            for key in param:
                help=help+([param[key]],)
            #print("help=", help)
            beta2 = np.concatenate(help, axis=0).transpose()
            #print('beta=', beta)
            print("beta2=", beta2)
            print("type=", type(beta2))

            if order_flag == 0:

                q = len(box)-1
                m = 0
                while m < n:
                    q = q+1
                    nodes, weights, m = SmolyakRule(len(box), q, typedistribution, box, beta2, is_loguniform)

                #q = q -1
                nodes = au.array2tuple(nodes,1) 

            else:
                nodes, weights, m = SmolyakRule(len(box), n+1, typedistribution, box, beta2, is_loguniform)
                q = n
                nodes = au.array2tuple(nodes,1)  
            #print("final nodes=", nodes)
            return nodes, weights

        if (typeGrid == 3):
            
            print("In typegrid==3 of DrawFrom")
            
            #beta_a = [self.kwargs['a'] for _ in range(len(box))] # gli dice che deve creare una lista di 9 per il primo parametro della beta
            #beta_b = [self.kwargs['b'] for _ in range(len(box))]
            #beta = np.concatenate(([beta_a],[beta_b]), axis=0).transpose()
            #print("beta_a, beta_b=", beta_a, beta_b)
            #print("([beta_a],[beta_b])=", ([beta_a],[beta_b]))

            param={}
            for key in self.kwargs.keys():
                param[key]= [self.kwargs[key] for _ in range(len(box))] # gli dice che deve creare una lista di 9 per il primo parametro della beta     
            print('param=', param)
            for key in param.keys():
                print("param[", key, "]=", param[key])       
            help = ()
            for key in param:
                help=help+([param[key]],)
            #print("help=", help)
            beta2 = np.concatenate(help, axis=0).transpose()
            #print('beta=', beta)
            print("beta2=", beta2)
            print("type=", type(beta2))

            assert(len(box)==len(beta2))
            nodes, weights = au.HaltonRule(len(box), n, typedistribution, box, beta2, is_loguniform)
            nodes = au.array2tuple(nodes,1)
            print("final nodes", nodes)
            return nodes, weights
    
