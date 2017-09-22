# Copyright (C) 2015-2017 by the RBniCS authors
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

from rbnics.utils.decorators import overload

def evaluate(backend, wrapping, online_backend, online_wrapping):
    class _Evaluate(object):
        @overload(backend.Function.Type(), (backend.ReducedMesh, backend.ReducedVertices))
        def __call__(self, function, at):
            return wrapping.evaluate_sparse_function_at_dofs(function, at.get_dofs_list())
        
        @overload(backend.FunctionsList, (backend.ReducedMesh, backend.ReducedVertices))    
        def __call__(self, functions_list, at):
            out_size = len(at.get_dofs_list())
            out = online_backend.OnlineMatrix(out_size, out_size)
            for (j, fun_j) in enumerate(functions_list):
                evaluate_fun_j = self.__call__(fun_j, at)
                for (i, out_ij) in enumerate(evaluate_fun_j):
                    out[i, j] = out_ij
            return out
        
        @overload(backend.ParametrizedExpressionFactory, None)
        def __call__(self, parametrized_expression, at):
            return wrapping.expression_on_truth_mesh(parametrized_expression)
        
        @overload(backend.ParametrizedExpressionFactory, (backend.ReducedMesh, backend.ReducedVertices))    
        def __call__(self, parametrized_expression, at):
            # Efficient version, interpolating only on the reduced mesh
            interpolated_expression = wrapping.expression_on_reduced_mesh(parametrized_expression, at)
            return wrapping.evaluate_sparse_function_at_dofs(interpolated_expression, at.get_reduced_dofs_list())
            """
            # Inefficient version, interpolating on the entire high fidelity mesh
            interpolated_expression = wrapping.expression_on_truth_mesh(parametrized_expression)
            return wrapping.evaluate_sparse_function_at_dofs(interpolated_expression, at.get_dofs_list())
            """
        
        @overload(backend.Matrix.Type(), backend.ReducedMesh)    
        def __call__(self, matrix, at):
            return wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs(matrix, at.get_dofs_list())
        
        @overload(backend.Vector.Type(), backend.ReducedMesh)    
        def __call__(self, vector, at):
            return wrapping.evaluate_sparse_vector_at_dofs(vector, at.get_dofs_list())
        
        @overload(backend.TensorsList, backend.ReducedMesh)    
        def __call__(self, tensors_list, at):
            out_size = len(at.get_dofs_list())
            out = online_backend.OnlineMatrix(out_size, out_size)
            for (j, tensor_j) in enumerate(tensors_list):
                evaluate_tensor_j = self.__call__(tensor_j, at)
                for (i, out_ij) in enumerate(evaluate_tensor_j):
                    out[i, j] = out_ij
            return out
        
        @overload(backend.ParametrizedTensorFactory, None)    
        def __call__(self, parametrized_tensor, at):
            (assembled_form, _) = wrapping.form_on_truth_function_space(parametrized_tensor)
            return assembled_form
        
        @overload(backend.ParametrizedTensorFactory, backend.ReducedMesh)    
        def __call__(self, parametrized_tensor, at):
            # Efficient version, assemblying only on the reduced mesh
            (assembled_form, form_rank) = wrapping.form_on_reduced_function_space(parametrized_tensor, at)
            assert form_rank in (1, 2)
            if form_rank is 2:
                return wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs(assembled_form, at.get_reduced_dofs_list())
            elif form_rank is 1:
                return wrapping.evaluate_sparse_vector_at_dofs(assembled_form, at.get_reduced_dofs_list())
            else: # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid form rank")
            """
            # Inefficient version, assemblying on the entire high fidelity mesh
            (assembled_form, form_rank) = wrapping.form_on_truth_function_space(parametrized_tensor)
            assert form_rank in (1, 2)
            if form_rank is 2:
                return wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs(assembled_form, at.get_dofs_list())
            elif form_rank is 1:
                return wrapping.evaluate_sparse_vector_at_dofs(assembled_form, at.get_dofs_list())
            else: # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid form rank")
            """
    return _Evaluate()
