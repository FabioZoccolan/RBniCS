# Copyright (C) 2015-2018 by the RBniCS authors
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

from rbnics.backends.basic.wrapping import DelayedBasisFunctionsMatrix, DelayedLinearSolver, DelayedProduct, DelayedSum, NonAffineExpansionStorageItem
from rbnics.backends.common.affine_expansion_storage import AffineExpansionStorage
from rbnics.utils.decorators import array_of, backend_for, list_of, overload, ThetaType

# product function to assemble truth/reduced affine expansions. To be used in combination with sum,
# even though this one actually carries out both the sum and the product!
@backend_for("common", inputs=(ThetaType, (AffineExpansionStorage, array_of(DelayedBasisFunctionsMatrix), array_of(DelayedLinearSolver)), ThetaType + (None,)))
def product(thetas, operators, thetas2=None):
    return _product(thetas, operators, thetas2)
    
@overload
def _product(thetas: ThetaType, operators: AffineExpansionStorage, thetas2: None):
    output = 0.
    assert len(thetas) == len(operators)
    for (theta, operator) in zip(thetas, operators):
        output += theta*operator
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: AffineExpansionStorage, thetas2: ThetaType):
    output = 0.
    # no checks here on the first dimension of operators should be equal to len(thetas), and
    # similarly that the second dimension should be equal to len(thetas2), because the
    # current operator interface does not provide a 2D len method
    for i, theta_i in enumerate(thetas):
        for j, theta2_j in enumerate(thetas2):
            output += theta_i*operators[i, j]*theta2_j
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: (array_of(DelayedLinearSolver), list_of(DelayedLinearSolver)), thetas2: None):
    output = None
    assert len(thetas) == len(operators)
    for (theta, operator) in zip(thetas, operators):
        assert isinstance(operator._rhs, (DelayedProduct, NonAffineExpansionStorageItem))
        if isinstance(operator._rhs, NonAffineExpansionStorageItem):
            rhs = theta*operator._rhs
        elif isinstance(operator._rhs, DelayedProduct):
            assert len(operator._rhs._args) is 3
            assert operator._rhs._args[0] == -1
            assert isinstance(operator._rhs._args[1], NonAffineExpansionStorageItem)
            rhs = (theta*operator._rhs._args[0])*operator._rhs._args[1]*operator._rhs._args[2]
        else:
            raise TypeError("Invalid rhs")
        if output is None:
            output = DelayedLinearSolver(operator._lhs, operator._solution, DelayedSum(rhs), operator._bcs)
        else:
            assert output._lhs is operator._lhs
            assert output._solution is operator._solution
            output._rhs += rhs
            assert output._bcs is operator._bcs
    output = output.solve()
    return ProductOutput(output)
    
@overload
def _product(thetas: ThetaType, operators: array_of(DelayedBasisFunctionsMatrix), thetas2: None):
    from rbnics.backends import BasisFunctionsMatrix
    space = operators[0].space
    assert all([op.space == space for op in operators])
    components_name = operators[0]._components_name
    assert all([op._components_name == components_name for op in operators])
    output = BasisFunctionsMatrix(space)
    output.init(components_name)
    for component_name in components_name:
        operator_memory_over_basis_functions_index = None # list (over basis functions index) of list (over theta)
        for operator in operators:
            operator_memory = operator._enrich_memory[component_name] # list (over basis functions index) for current theta
            if operator_memory_over_basis_functions_index is None:
                operator_memory_over_basis_functions_index = [list() for _ in operator_memory]
            assert len(operator_memory_over_basis_functions_index) is len(operator_memory)
            for (basis_functions_index, delayed_function) in enumerate(operator_memory):
                operator_memory_over_basis_functions_index[basis_functions_index].append(delayed_function)
        for delayed_functions_over_theta in operator_memory_over_basis_functions_index:
            output.enrich(_product(thetas, delayed_functions_over_theta, None).sum_product_return_value, component=component_name if len(components_name) > 1 else None)
    return ProductOutput(output)
    
# Auxiliary class to signal to the sum() function that it is dealing with an output of the product() method
class ProductOutput(object):
    def __init__(self, sum_product_return_value):
        self.sum_product_return_value = sum_product_return_value
