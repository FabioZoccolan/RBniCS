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
from rbnics.sampling.distributions.equispaced_distribution import EquispacedDistribution

class CompositeDistribution(Distribution):
    def __init__(self, distributions, typedistribution, is_loguniform):
        print("CompositeDistribution has been initialized. Parameters are")
        self.is_loguniform = is_loguniform
        self.distributions = distributions
        self.typedistribution = typedistribution
        # Create a dict from scalar distribution to component
        self.distribution_to_components = dict()
        print("\tdistributions is ", distributions) # \t is a tab
        print("\ttypedistribution is ", typedistribution)
        print("\ttype(distributions)=",type(distributions))
        for (p, distribution) in enumerate(self.distributions): #The enumerate() method adds a counter to an iterable and returns it (the enumerate object).
            assert isinstance(distribution, Distribution)
            if distribution not in self.distribution_to_components:
                self.distribution_to_components[distribution] = list()
            self.distribution_to_components[distribution].append(p)

    def sample(self, box, n, sparse_flag=False, order_flag=0):
        print("i am compositedistribution.sample")
        """genera n realizzazioni nel box per la variabile che vogliamo
         ridà un dizionario che contiene n liste di realizzazioni         
         """
        print("ComDist.sample::is_loguniform=", self.is_loguniform)
       
        # Divide box among the different distributions
        distribution_to_sub_box = dict() #dict() function creates a dictionary.
        for (distribution, components) in self.distribution_to_components.items():
            print("CompDist.sample::distribution, components", distribution, components)
            print("these components have boxes:")
            for p in components:
                print("component ", p, ":", box[p])
            distribution_to_sub_box[distribution] = [box[p] for p in components] # dict that contains the different ranges 
        # Prepare a dict that will store the map from components to subset sub_set
        components_to_sub_set = dict()
        # Consider first equispaced distributions, because they may change the value of n
        #weights = []
        for (distribution, sub_box) in distribution_to_sub_box.items():
            print("isinstance(distribution, EquispacedDistribution):", isinstance(distribution, EquispacedDistribution))
            if isinstance(distribution, EquispacedDistribution):
                sub_box = distribution_to_sub_box[distribution]
                if sparse_flag:
                    sub_set, weights = distribution.sample(sub_box, n, self.typedistribution, sparse_flag, order_flag, self.is_loguniform) 
                else: 
                    sub_set = distribution.sample(sub_box, n, self.typedistribution, sparse_flag, order_flag, self.is_loguniform)
                n = len(sub_set) # may be greater or equal than the one originally provided
                components = self.distribution_to_components[distribution]
                components_to_sub_set[tuple(components)] = sub_set
        assert len(components_to_sub_set) in (0, 1)
        # ... and the consider all the remaining distributions
        for (distribution, sub_box) in distribution_to_sub_box.items():
            print('CompDist.sample::now calling .sample of distribution', distribution, 'with support', sub_box)
            if not isinstance(distribution, EquispacedDistribution):
                components = self.distribution_to_components[distribution]
                if sparse_flag:
                    print("sparse_flag=", sparse_flag)
                    print("Now we use:: components_to_sub_set[tuple(components)], weights = distribution.sample(sub_box, n, self.typedistribution, sparse_flag, order_flag, self.is_loguniform)")
                    components_to_sub_set[tuple(components)], weights = distribution.sample(sub_box, n, self.typedistribution, sparse_flag, order_flag, self.is_loguniform)
                else:
                    print("sparse_flag=", sparse_flag)
                    components_to_sub_set[tuple(components)]= distribution.sample(sub_box, n, self.typedistribution, sparse_flag, order_flag, self.is_loguniform)
        # Prepare a list that will store the set [mu_1, ... mu_n] ...
        if (sparse_flag):
            set_as_list = [[None]*len(box) for _ in range(len(weights))]
        else:
            set_as_list = [[None]*len(box) for _ in range(n)]
        for (components, sub_set) in components_to_sub_set.items():
            if not sparse_flag: 
                assert len(sub_set) == n
            for (index, sub_mu) in enumerate(sub_set): #A list of tuples, the jth tuple contains the jth sample points for the mu_p in components
                assert len(components) == len(sub_mu)
                for (p, sub_mu_p) in zip(components, sub_mu):
                    set_as_list[index][p] = sub_mu_p
        # ... and convert each mu to a tuple
        set_ = [tuple(mu) for mu in set_as_list]
        if sparse_flag:
            return set_, weights
        else:
            return set_
