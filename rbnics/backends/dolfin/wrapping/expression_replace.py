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

# This file contains a copy of UFL replace with a customized Replacer,
# which also handles Indexed and ListTensor cases

from ufl.algorithms.map_integrands import map_integrand_dags
from ufl.algorithms.analysis import has_exact_type
from ufl.classes import CoefficientDerivative
from ufl.constantvalue import as_ufl
from ufl.core.multiindex import FixedIndex, Index, MultiIndex
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_domains
from ufl.indexed import Indexed
from ufl.log import error
from ufl.tensors import ComponentTensor, ListTensor
from dolfin import Function, split

def expression_replace(expression, replacements):
    replaced_expression = replace(expression, replacements)
    replaced_expression_domains = extract_domains(replaced_expression)
    assert len(replaced_expression_domains) in (0, 1)
    expression_domains = extract_domains(expression)
    assert len(expression_domains) in (0, 1)
    assert len(expression_domains) == len(replaced_expression_domains)
    if len(expression_domains) is 1:
        assert replaced_expression_domains[0] is not expression_domains[0]
    return replaced_expression

class Replacer(MultiFunction):
    def __init__(self, mapping):
        MultiFunction.__init__(self)
        for k in mapping:
            if (
                not k._ufl_is_terminal_
                    and
                not isinstance(k, (Indexed, ListTensor))
            ):
                error("This implementation can only replace Terminal objects or non terminal Indexed and ListTensor objects.")
        if not all(k.ufl_shape == v.ufl_shape for k, v in mapping.iteritems()):
            error("Replacement expressions must have the same shape as what they replace.")
        # Prepare the mapping extracting each Indexed from any ListTensor
        self._mapping = dict()
        for (k, v) in mapping.iteritems():
            if isinstance(k, ListTensor):
                split_k = k.ufl_operands
                split_v = split(v)
                assert len(split_k) == len(split_v)
                for (sub_k, sub_v) in zip(split_k, split_v):
                    self._mapping[sub_k] = sub_v
            else:
                self._mapping[k] = v

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, o):
        e = self._mapping.get(o)
        if e is None:
            return o
        else:
            return e
            
    def indexed(self, o, *dops):
        if o in self._mapping:
            return self._mapping[o]
        else:
            assert len(o.ufl_operands) == 2
            assert isinstance(o.ufl_operands[1], MultiIndex)
            if o.ufl_operands[0] in self._mapping:
                replaced_ufl_operand_0 = self._mapping[o.ufl_operands[0]]
            else:
                replaced_ufl_operand_0 = map_integrand_dags(self, o.ufl_operands[0])
            return Indexed(replaced_ufl_operand_0, o.ufl_operands[1])
    
    def list_tensor(self, o, *dops):
        assert o not in self._mapping
        replaced_ufl_operands = list()
        for ufl_operand in o.ufl_operands:
            if ufl_operand in self._mapping:
                replaced_ufl_operands.append(self._mapping[ufl_operand])
            else:
                replaced_ufl_operands.append(map_integrand_dags(self, ufl_operand))
        return ListTensor(*replaced_ufl_operands)
        
    def component_tensor(self, o, *dops):
        assert o not in self._mapping
        assert len(o.ufl_operands) == 2
        assert isinstance(o.ufl_operands[1], MultiIndex)
        if o.ufl_operands[0] in self._mapping:
            replaced_ufl_operand_0 = self._mapping[o.ufl_operands[0]]
        else:
            replaced_ufl_operand_0 = map_integrand_dags(self, o.ufl_operands[0])
        return ComponentTensor(replaced_ufl_operand_0, o.ufl_operands[1])

    def coefficient_derivative(self, o):
        error("Derivatives should be applied before executing replace.")
        
def replace(e, mapping):
    """Replace objects in expression.

    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.iteritems())
    
    # We have expanded derivative evaluation in ParametrizedTensorFactory
    assert not has_exact_type(e, CoefficientDerivative)
    
    return map_integrand_dags(Replacer(mapping2), e)
