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
from numpy import abs, cumsum as compute_retained_energy, isclose, float, loadtxt, sum as compute_total_energy
from rbnics.utils.io import ExportableList

# Class containing the implementation of the POD
def ProperOrthogonalDecompositionBase(backend, wrapping, online_backend, online_wrapping, ParentProperOrthogonalDecomposition, SnapshotsContainerType, BasisContainerType):
    
    class _ProperOrthogonalDecompositionBase(ParentProperOrthogonalDecomposition):

        def __init__(self, space, inner_product, *args):
            print("PODbase initialized")
            self.inner_product = inner_product
            self.space = space
            self.args = args
            self.backend = backend

            # Declare a matrix to store the snapshots
            self.snapshots_matrix = SnapshotsContainerType(self.space, *args)
            print("PODbase.__init__::Our snapshot matrix is", self.snapshots_matrix)
            # Declare a list to store eigenvalues
            self.eigenvalues = ExportableList("text")
            self.retained_energy = ExportableList("text")
            
        def clear(self):
            self.snapshots_matrix.clear()
            self.eigenvalues = ExportableList("text")
            self.retained_energy = ExportableList("text")
            
        # No implementation is provided for store_snapshot, because
        # it has different interface for the standard POD and
        # the tensor one.
        
        def weighted_apply(self, Nmax, tol, name):
            print("In PODbase.weighted_apply")   
            inner_product = self.inner_product
            snapshots_matrix = self.snapshots_matrix
            transpose = backend.transpose
            ## Load the weights
            print("PODbase.weighted_apply::loading weights")
            weights = loadtxt(name + "/training_set/training_set_weight.txt", dtype= 'str', delimiter = ', ')
            print("PODbase.weighted_apply::finished loading weights, they are", weights)
            weights[0] = weights[0][1:]
            weights[len(weights)-1] = weights[len(weights)-1][:-1]
            weights = weights.astype(float)

            if inner_product is not None:
                    correlation = transpose(snapshots_matrix)*inner_product*snapshots_matrix
            else:
                    correlation = transpose(snapshots_matrix)*snapshots_matrix

            print("PODbase.weighted_apply::The correlation matrix of size", len(self.snapshots_matrix),"**2 BEFORE WEIGHING:", correlation)
            
            
            #assert len(correlation) == len(self.snapshots_matrix) and len(correlation) == len(weights)
            for i in range(Nmax):
                for j in range(Nmax):
                    correlation[i,j] = correlation[i,j] * weights[i]
            #for i in range(Nmax):
            #    for j in range(Nmax):
            #        correlation[i,j] = weights[i] * correlation[i,j] * weights[j]
            

            print("PODbase.weighted_apply::The correlation matrix of size", len(self.snapshots_matrix),"**2 AFTER WEIGHING:", correlation)
            #print("The correlation matrix of size", len(self.snapshots_matrix),"**2:")
            basis_functions = BasisContainerType(self.space, *self.args)

            eigensolver = online_backend.OnlineEigenSolver(basis_functions, correlation)
            parameters = {
                "problem_type": "hermitian",
                "spectrum": "largest real"
            }
            eigensolver.set_parameters(parameters)
            eigensolver.solve()
            
            Neigs = len(self.snapshots_matrix)
            Nmax = min(Nmax, Neigs)
            print("PODbase.weighted_apply::Nmax=min(self.Nmax, Neigs)=", Nmax)
            assert len(self.eigenvalues) == 0
            for i in range(Neigs):
                (eig_i_real, eig_i_complex) = eigensolver.get_eigenvalue(i)
                assert isclose(eig_i_complex, 0.)
                self.eigenvalues.append(eig_i_real)
            
            total_energy = compute_total_energy([abs(e) for e in self.eigenvalues])
            retained_energy = compute_retained_energy([abs(e) for e in self.eigenvalues])
            assert len(self.retained_energy) == 0
            if total_energy > 0.:
                self.retained_energy.extend([retained_energy_i/total_energy for retained_energy_i in retained_energy])
            else:
                self.retained_energy.extend([1. for _ in range(Neigs)]) # trivial case, all snapshots are zero
            
            eigenvectors = list()
            for N in range(Nmax):
                (eigvector, _) = eigensolver.get_eigenvector(N)
                eigenvectors.append(eigvector)
                b = self.snapshots_matrix*eigvector
                if inner_product is not None:
                    norm_b = sqrt(transpose(b)*inner_product*b)
                else:
                    norm_b = sqrt(transpose(b)*b)
                if norm_b != 0.:
                    b /= norm_b
                basis_functions.enrich(b)
                if self.retained_energy[N] > 1. - tol:
                    break
            N += 1
            print("PODbase.weighted_apply::Completed compute retained energy")
            return (self.eigenvalues[:N], eigenvectors, basis_functions, N)        
            print("Finished PODbase.weighted_apply, basis_functions=", basis_functions)
        
        def apply(self, Nmax, tol):
            print("In PODbase.apply")
          
            inner_product = self.inner_product
            snapshots_matrix = self.snapshots_matrix
            transpose = backend.transpose
            print("PODbase.apply::inner_product and snapshots_matrix", inner_product, snapshots_matrix)
            
            if inner_product is not None:
                correlation = transpose(snapshots_matrix)*inner_product*snapshots_matrix
            else:
                correlation = transpose(snapshots_matrix)*snapshots_matrix
            #print("The correlation matrix of size", len(self.snapshots_matrix),"**2:", correlation)
            print("PODbase.apply::The correlation matrix of size", len(self.snapshots_matrix),"**2:", correlation)
            basis_functions = BasisContainerType(self.space, *self.args)
            
            eigensolver = online_backend.OnlineEigenSolver(basis_functions, correlation)
            parameters = {
                "problem_type": "hermitian",
                "spectrum": "largest real"
            }
            eigensolver.set_parameters(parameters)
            eigensolver.solve()
            
            Neigs = len(self.snapshots_matrix)
            Nmax = min(Nmax, Neigs)
            print("PODbase.apply::Nmax=min(self.Nmax, Neigs)=", Nmax)
            assert len(self.eigenvalues) == 0
            for i in range(Neigs):
                (eig_i_real, eig_i_complex) = eigensolver.get_eigenvalue(i)
                assert isclose(eig_i_complex, 0.)
                self.eigenvalues.append(eig_i_real)
            
            total_energy = compute_total_energy([abs(e) for e in self.eigenvalues])
            retained_energy = compute_retained_energy([abs(e) for e in self.eigenvalues])
            assert len(self.retained_energy) == 0
            if total_energy > 0.:
                self.retained_energy.extend([retained_energy_i/total_energy for retained_energy_i in retained_energy])
            else:
                self.retained_energy.extend([1. for _ in range(Neigs)]) # trivial case, all snapshots are zero
            
            eigenvectors = list()
            for N in range(Nmax):
                (eigvector, _) = eigensolver.get_eigenvector(N)
                eigenvectors.append(eigvector)
                b = self.snapshots_matrix*eigvector
                if inner_product is not None:
                    norm_b = sqrt(transpose(b)*inner_product*b)
                else:
                    norm_b = sqrt(transpose(b)*b)
                if norm_b != 0.:
                    b /= norm_b
                basis_functions.enrich(b)
                if self.retained_energy[N] > 1. - tol:
                    break
            N += 1
            print("PODbase.apply::Completed compute retained energy")
            return (self.eigenvalues[:N], eigenvectors, basis_functions, N)
            print("Finished PODbase.apply, basis_functions=", basis_functions)
                
        def print_eigenvalues(self, N=None):
            print("IN PODbase.print_eigenvalues")
            if N is None:
                N = len(self.snapshots_matrix)
            for i in range(N):
                print("lambda_" + str(i) + " = " + str(self.eigenvalues[i]))
            print("Completed PODbase.print_eigenvalues")

        def save_eigenvalues_file(self, output_directory, eigenvalues_file):
            print("PODbase.save_eigehvalues_file::Saving eigenvalues")
            self.eigenvalues.save(output_directory, eigenvalues_file)
            
        def save_retained_energy_file(self, output_directory, retained_energy_file):
            print("PODbase.save_retained_energy_file::Saving retained energy")
            self.retained_energy.save(output_directory, retained_energy_file)

    return _ProperOrthogonalDecompositionBase
