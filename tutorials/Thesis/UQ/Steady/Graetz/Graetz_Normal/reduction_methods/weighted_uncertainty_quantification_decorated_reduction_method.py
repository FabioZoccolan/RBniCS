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

from math import sqrt
from logging import DEBUG, getLogger
from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from problems import WeightedUncertaintyQuantification

#logger = getLogger("tutorials/10_weighted_uq/reduction_methods/weighted_uncertainty_quantification_decorated_reduction_method.py")
logger = getLogger("tutorials/Tesi/02_elliptic_ocp_weighted_poisson/reduction_methods/weighted_uncertainty_quantification_decorated_reduction_method.py")

@ReductionMethodDecoratorFor(WeightedUncertaintyQuantification)
def WeightedUncertaintyQuantificationDecoratedReductionMethod(EllipticOptimalControlReductionMethod_DerivedClass):
    
    @PreserveClassName
    class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base(EllipticOptimalControlReductionMethod_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            print("\nInitializing weighted_uncertainty_quantification_decorated_reduction_method\n")
            print("\nWEIGHTED UQ CLASS INITIALIZED\n")
            EllipticOptimalControlReductionMethod_DerivedClass.__init__(self, truth_problem, **kwargs)
            self.weight = None
            self.training_set_density = None
            
        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, weight=None, **kwargs):
            print("\n*Started initialize_training_set of WeightedUQ class\n")
            import_successful = EllipticOptimalControlReductionMethod_DerivedClass.initialize_training_set(self, ntrain, enable_import, sampling, **kwargs)
            print("\n**Finished initialize_training_set of WeightedUQ class\n")
            self.weight = weight
            return import_successful
            
        def _offline(self):
            # Initialize densities
            print("\n*WeightedUQ._offline::started\n")
            tranining_set_and_first_mu = [mu for mu in self.training_set]
            tranining_set_and_first_mu.append(self.truth_problem.mu)
            print("tranining_set_and_first_mu=", tranining_set_and_first_mu)
            if self.weight is not None:
                print("Weights were assigned directly in the tutorial")
                self.training_set_density = dict(zip(tranining_set_and_first_mu, self.weight.density(self.truth_problem.mu_range, tranining_set_and_first_mu)))
            else:
                self.training_set_density = {mu: 1. for mu in tranining_set_and_first_mu}
            
            # Call Parent method
            EllipticOptimalControlReductionMethod_DerivedClass._offline(self)
            print("**WeightedUQ._offline::finished\n")
            
    if hasattr(EllipticOptimalControlReductionMethod_DerivedClass, "greedy"): # RB reduction
        @PreserveClassName
        class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class(WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base):
            def _greedy(self):
                """
                It chooses the next parameter in the offline stage in a greedy fashion. Internal method.
                
                :return: max error estimator and the respective parameter.
                """
                
                def weight(mu):
                    return sqrt(self.training_set_density[mu])
                
                # Print some additional information on the consistency of the reduced basis
                print("absolute error for current mu =", self.reduced_problem.compute_error())
                print("absolute (weighted) error estimator for current mu =", weight(self.truth_problem.mu)*self.reduced_problem.estimate_error())
                print("absolute non-weighted error estimator for current mu =", self.reduced_problem.estimate_error())
                
                # Carry out the actual greedy search
                def solve_and_estimate_error(mu):
                    self.reduced_problem.set_mu(mu)
                    self.reduced_problem.solve()
                    error_estimator = self.reduced_problem.estimate_error()
                    weighted_error_estimator = weight(mu)*error_estimator
                    logger.log(DEBUG, "(Weighted) error estimator for mu = " + str(mu) + " is " + str(weighted_error_estimator))
                    logger.log(DEBUG, "Non-weighted error estimator for mu = " + str(mu) + " is " + str(error_estimator))
                    return weighted_error_estimator
                
                print("find next mu")
                return self.training_set.max(solve_and_estimate_error)
    else: # POD reduction
        @PreserveClassName
        class WeightedUncertaintyQuantificationDecoratedReductionMethod_Class(WeightedUncertaintyQuantificationDecoratedReductionMethod_Class_Base):
            def update_snapshots_matrix(self, snapshot):
                def weight(mu):
                    ERROR_YOU_SHOULD_NOT_BE_IN_THIS_FUNCTION
                    return sqrt(self.training_set_density[mu])
                self.POD["y"].store_snapshot(snapshot, component="y")
                self.POD["u"].store_snapshot(snapshot, component="u")
                self.POD["p"].store_snapshot(snapshot, component="p")
    # return value (a class) for the decorator
    return WeightedUncertaintyQuantificationDecoratedReductionMethod_Class
