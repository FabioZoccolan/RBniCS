# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.sampling.distributions.distribution import Distribution
from rbnics.sampling.distributions.equispaced_distribution import EquispacedDistribution


class CompositeDistribution(Distribution):
    def __init__(self, distributions, typedistribution, is_loguniform):
    
        print("\nCompositeDistribution has been initialized (__init__). Parameters are:") #added
        
        self.is_loguniform = is_loguniform #added
        self.distributions = distributions
        self.typedistribution = typedistribution #added
        
        # Create a dictionary from scalar distribution to component
        self.distribution_to_components = dict()
        
        print("\tdistributions is ", distributions) #added
        print("\ttypedistribution is ", typedistribution) #added
        print("\ttype(distributions)=",type(distributions)) #added
        
        for (p, distribution) in enumerate(self.distributions):
            assert isinstance(distribution, Distribution) #isinstance specifies if an object is of a particular type or not
            if distribution not in self.distribution_to_components:
                self.distribution_to_components[distribution] = list()
            self.distribution_to_components[distribution].append(p)
        print("Finished CompositeDistribution (__init__)\n")


    def sample(self, box, n, sparse_flag=False, order_flag=0): #added
    
        print("ComDist.sample::is_loguniform=", self.is_loguniform) #ComDist = Composite Distribution #added
        print("I am compositedistribution.sample") #added
        """genera n realizzazioni nel box per la variabile che vogliamo
         ridà un dizionario che contiene n liste di realizzazioni         
         """
        # Divide box among the different distributions
        distribution_to_sub_box = dict()
        
        for (distribution, components) in self.distribution_to_components.items():
        
            print("CompDist.sample::distribution, components", distribution, components) #added
            print("these components have boxes:") #added
            for p in components: #added
                print("component ", p, ":", box[p]) #added
            distribution_to_sub_box[distribution] = [box[p] for p in components] # dict that contains the different ranges 
            
        # Prepare a dict that will store the map from components to subset sub_set
        components_to_sub_set = dict()
        
        # Consider first equispaced distributions, because they may change the value of n
        #weights = []
        for (distribution, sub_box) in distribution_to_sub_box.items():
            if isinstance(distribution, EquispacedDistribution):
                sub_box = distribution_to_sub_box[distribution]
                if sparse_flag: #added
                    sub_set, weights = distribution.sample(sub_box, n, self.typedistribution, sparse_flag, order_flag, self.is_loguniform)  #added
                else:  #added
                    sub_set = distribution.sample(sub_box, n)
                n = len(sub_set)  # may be greater or equal than the one originally provided
                components = self.distribution_to_components[distribution]
                components_to_sub_set[tuple(components)] = sub_set
                
        assert len(components_to_sub_set) in (0, 1)
        
        # ... and the consider all the remaining distributions
        for (distribution, sub_box) in distribution_to_sub_box.items():
             print('CompDist.sample::now calling .sample of distribution', distribution, 'with support', sub_box) #added
             if not isinstance(distribution, EquispacedDistribution):
                components = self.distribution_to_components[distribution]
                if sparse_flag:
                    print("sparse_flag=", sparse_flag)
                    components_to_sub_set[tuple(components)], weights = distribution.sample(sub_box, n, self.typedistribution, sparse_flag, order_flag, self.is_loguniform)
                else:
                    print("sparse_flag=", sparse_flag)
                    components_to_sub_set[tuple(components)] = distribution.sample(sub_box, n, self.typedistribution, sparse_flag, order_flag, self.is_loguniform)
                    
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
            
        
