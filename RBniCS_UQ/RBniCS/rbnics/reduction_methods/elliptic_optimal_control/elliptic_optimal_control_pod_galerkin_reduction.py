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


from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearPODGalerkinReduction
from rbnics.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_reduction_method import EllipticOptimalControlReductionMethod
from os.path import isfile

EllipticOptimalControlPODGalerkinReduction_Base = LinearPODGalerkinReduction(EllipticOptimalControlReductionMethod(DifferentialProblemReductionMethod))

@ReductionMethodFor(EllipticOptimalControlProblem, "PODGalerkin")
class EllipticOptimalControlPODGalerkinReduction(EllipticOptimalControlPODGalerkinReduction_Base):
    
    # Compute basis functions performing POD: overridden to handle aggregated spaces
    def compute_basis_functions(self):
        # Carry out POD
        basis_functions = dict()
        N = dict()
        print("self.truth_problem.components", self.truth_problem.components)
        for component in self.truth_problem.components:
            print("# POD for component", component)
            POD = self.POD[component]
            assert self.tol[component] == 0. # TODO first negelect tolerances, then compute the max of N for each aggregated pair
            print("EllipticOCPPODGRED.compute_basis_functions::self.Nmax, self.tol=", self.Nmax, self.tol)
            if isfile(self.truth_problem.name() + "/training_set/training_set_weight.txt"):
                print("EllipticOCPPODGRED.compute_basis_functions::WEIGHT FILE FOUND-->calling POD.Weighted_apply")
                (_, _, basis_functions[component], N[component]) = POD.weighted_apply(self.Nmax, self.tol[component], self.truth_problem.name())
            else:
                print("EllipticOCPPODGRED.compute_basis_functions::NO WEIGHT FILE FOUND-->calling POD.apply")
                (_, _, basis_functions[component], N[component]) = POD.apply(self.Nmax, self.tol[component])
            print("EllipticOCPPODGRED.compute_basis_functions::basis_functions and N are", basis_functions, N)
            POD.print_eigenvalues(N[component])
            POD.save_eigenvalues_file(self.folder["post_processing"], "eigs_" + component)
            POD.save_retained_energy_file(self.folder["post_processing"], "retained_energy_" + component)
        
        # Store POD modes related to control as usual
        self.reduced_problem.basis_functions.enrich(basis_functions["u"], component="u")
        self.reduced_problem.N["u"] += N["u"]
        print("self.reduced_problem.N["u"] += N["u"]", N["u"])
        
        # Aggregate POD modes related to state and adjoint
        for component_to in ("y", "p"):
            for i in range(self.Nmax): # TODO should have been N[component_from], but cannot switch the next line
                for component_from in ("y", "p"):
                    #print("I:component_from", i, ":", component_from)
                    #print("basis_functions[component_from][i]", basis_functions[component_from][i])
                    #print("self.reduced_problem.basis_functions.enrich(basis_functions[component_from][i], component={component_from: component_to}")
                    self.reduced_problem.basis_functions.enrich(basis_functions[component_from][i], component={component_from: component_to})
                self.reduced_problem.N[component_to] += 2
        #print( N[component_to], ": ", self.reduced_problem.N[component_to], N[component_to], ": ", self.reduced_problem.N[component_to])
        
        # Save
        self.reduced_problem.basis_functions.save(self.reduced_problem.folder["basis"], "basis")