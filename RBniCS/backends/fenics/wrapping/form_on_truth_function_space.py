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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl.algorithms.traversal import iter_expressions
from ufl.corealg.traversal import traverse_unique_terminals
from dolfin import assign, Function
from RBniCS.backends.fenics.wrapping.function_from_subfunction_if_any import function_from_subfunction_if_any
from RBniCS.utils.decorators import get_problem_from_solution, get_reduced_problem_from_problem
from RBniCS.eim.utils.decorators import get_EIM_approximation_from_parametrized_expression

def form_on_truth_function_space(form_wrapper):
    form = form_wrapper._form
    EIM_approximation = get_EIM_approximation_from_parametrized_expression(form_wrapper)
    
    if form not in form_on_truth_function_space__reduced_problem_to_truth_solution_cache:
        visited = list()
        reduced_problem_to_truth_solution = dict()
        
        # Look for terminals on truth mesh
        for integral in form.integrals():
            for expression in iter_expressions(integral):
                for node in traverse_unique_terminals(expression):
                    node = function_from_subfunction_if_any(node)
                    if node in visited:
                        continue
                    # ... problem solutions related to nonlinear terms
                    elif isinstance(node, Function):
                        truth_problem = get_problem_from_solution(node)
                        reduced_problem = get_reduced_problem_from_problem(truth_problem)
                        reduced_problem_to_truth_solution[reduced_problem] = node
                        visited.append(node)
        
        # Cache the resulting dicts
        form_on_truth_function_space__reduced_problem_to_truth_solution_cache[form] = reduced_problem_to_truth_solution
        
    # Extract from cache
    reduced_problem_to_truth_solution = form_on_truth_function_space__reduced_problem_to_truth_solution_cache[form]
        
    # Solve reduced problem associated to nonlinear terms
    for (reduced_problem, truth_solution) in reduced_problem_to_truth_solution.iteritems():
        reduced_problem.set_mu(EIM_approximation.mu)
        reduced_solution = reduced_problem.solve()
        assign(truth_solution, reduced_problem.Z[:reduced_solution.N]*reduced_solution)
    
    return form
    
form_on_truth_function_space__reduced_problem_to_truth_solution_cache = dict()