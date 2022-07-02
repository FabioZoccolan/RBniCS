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

import operator # to find closest parameters
from math import sqrt
from mpi4py.MPI import COMM_WORLD
from numpy import zeros as array
from numpy import argmax
from rbnics.sampling.distributions import CompositeDistribution, UniformDistribution
from rbnics.utils.decorators import overload
from rbnics.utils.io import ExportableList
from rbnics.utils.mpi import parallel_io as parallel_generate, parallel_max

class ParameterSpaceSubset(ExportableList): # equivalent to a list of tuples
    def __init__(self):
        print("In ParameterSpaceSubset __init__")
        ExportableList.__init__(self, "text")
        self.mpi_comm = COMM_WORLD #mpi Message Passing Interface #
        self.distributed_max = True
        self.weight = ExportableList("text")
    @overload
    def __getitem__(self, key: int):
        return self._list[key]
        
    @overload
    def __getitem__(self, key: slice):
        output = ParameterSpaceSubset()
        output.distributed_max = self.distributed_max
        output._list = self._list[key]
        return output
    
    # Method for generation of parameter space subsets
    def generate(self, box, n, sampling=None, typeGrid=0, order_flag=False):
        print("In ParameterSpaceSubset.generate")
        print("ParameterSpaceSubset.generate::is_tuple(sampling)=", isinstance(sampling,tuple))
        if len(box) > 0:
            print("len(box) is:", len(box))
            print("isinstance(sampling, tuple):", isinstance(sampling, tuple))
            if sampling is None:
                sampling = UniformDistribution()
                print("Sampling is not set >> UniformDistribution()")
                # Sampling_Unspecified
            elif isinstance(sampling, tuple):
                assert len(sampling) == len(box)
                print("We set sampling = CompositeDistribution(sampling) - BEGIN")
                sampling = CompositeDistribution(sampling)
                print("We set sampling = CompositeDistribution(sampling) - END")
                
            def run_sampling(): # HERE WE GENERATE THE SAMPLE!!   
                print("\nIn run_sampling of ParameterSpaceSubset")
                print("ParameterSpaceSubset.generate::now calling compositeDistribution.sample")
                print("We call sampling.sample(box, n, typeGrid, order_flag) - BEGIN")
                return sampling.sample(box, n, typeGrid, order_flag)
                print("We call sampling.sample(box, n, typeGrid, order_flag) - END")
                
            print("ParameterSpaceSubset.generate::here typegrid is", typeGrid)
            if typeGrid != 0:
                print("I am now in param_space_subset.generate() for typegrid!=0, i.e. sthg which is not MonteCarlo")
                print("run_sampling=", run_sampling) #function just above
                self._list, weight = parallel_generate(run_sampling, self.mpi_comm) #import from rbnics.utils.parallel_io
                '''
                print("the weights have been assigned succesfully")
                import numpy as np
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                x=np.ones(len(self._list))
                y=np.ones(len(self._list))
                #z=np.ones(len(self._list))
                for i in range(len(self._list)):
                    x[i] = self._list[i][0]
                    y[i] = self._list[i][1]

                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.scatter(x,y,c='r', marker='o')

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                plt.show()  
                import sys
                sys.exit()
                '''
                self.weight.extend(weight)
            else:
                self._list = parallel_generate(run_sampling, self.mpi_comm)
                """import numpy as np
                import matplotlib.pyplot as plt
                x=np.ones(len(self._list))
                y=np.ones(len(self._list))
                #z=np.ones(len(self._list))
                for i in range(len(self._list)):
                    x[i] = self._list[i][0]
                    y[i] = self._list[i][1]
                fig = plt.figure()
                ax = fig.add_subplot(111)

                ax.scatter(x, y, c='r', marker='o')

                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                plt.show()
                import sys
                sys.exit() """
                
            # _list ha [i][j] dove sulla prima riga ha la tupla di parametri e sulla seconda quella di pesi associati
            print("\n\n\nnumero parametri:" ,len(self._list),"\n\n")
        else:
            for i in range(n):
                self._list.append(tuple())
        print("ParamaterSpaceSubset.generate::DONE")
        
    def max(self, generator, postprocessor=None):
        if postprocessor is None:
            def postprocessor(value):
                return value
        if self.distributed_max:
            local_list_indices = list(range(self.mpi_comm.rank, len(self._list), self.mpi_comm.size)) # start from index rank and take steps of length equal to size
        else:
            local_list_indices = list(range(len(self._list)))
        values = array(len(local_list_indices))
        values_with_postprocessing = array(len(local_list_indices))
        for i in range(len(local_list_indices)):
            values[i] = generator(self._list[local_list_indices[i]])
            values_with_postprocessing[i] = postprocessor(values[i])
        if self.distributed_max:
            local_i_max = argmax(values_with_postprocessing)
            local_value_max = values[local_i_max]
            (global_value_max, global_i_max) = parallel_max(local_value_max, local_list_indices[local_i_max], postprocessor, self.mpi_comm)
            assert isinstance(global_i_max, tuple)
            assert len(global_i_max) == 1
            global_i_max = global_i_max[0]
        else:
            global_i_max = argmax(values_with_postprocessing)
            global_value_max = values[global_i_max]
        return (global_value_max, global_i_max)
        
    def serialize_maximum_computations(self):
        assert self.distributed_max is True
        self.distributed_max = False
    
    def diff(self, other_set):
        output = ParameterSpaceSubset()
        output.distributed_max = self.distributed_max
        output._list = [mu for mu in self._list if mu not in other_set]
        return output
        
    # M parameters in this set closest to mu
    def closest(self, M, mu):
        assert M <= len(self)
        
        # Trivial case 1:
        if M == len(self):
            return self
            
        output = ParameterSpaceSubset()
        output.distributed_max = self.distributed_max
            
        # Trivial case 2:
        if M == 0:
            return output
        
        parameters_and_distances = list()
        for xi_i in self:
            distance = sqrt(sum([(x - y)**2 for (x, y) in zip(mu, xi_i)]))
            parameters_and_distances.append((xi_i, distance))
        parameters_and_distances.sort(key=operator.itemgetter(1))
        output._list = [xi_i for (xi_i, _) in parameters_and_distances[:M]]
        return output
        
    def save(self, directory, filename, typeGrid=0):
        print("There is something to save:", not ( not self.weight) )
        ExportableList.save(self, directory, filename)
        if typeGrid != 0:
            self.weight.save(directory, filename + "_weight")
    
    def load(self, directory, filename, typeGrid=0):
        ExportableList.load(self, directory, filename)
        if typeGrid != 0:
            self.weight.load(directory, filename + "_weight")
