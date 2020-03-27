# Copyright (C) 2015-2020 by the RBniCS authors
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

from rbnics.backends import LinearProblemWrapper, LinearSolver
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators

@RequiredBaseDecorators(None)
def LinearProblem(ParametrizedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class LinearProblem_Class(ParametrizedDifferentialProblem_DerivedClass):

        # Default initialization of members
        def __init__(self, V, **kwargs):
            # Call to parent
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)

            # Nonlinear solver parameters
            self._linear_solver_parameters = dict()

        class ProblemSolver(ParametrizedDifferentialProblem_DerivedClass.ProblemSolver, LinearProblemWrapper):
            def solve(self):
                problem = self.problem
                solver = LinearSolver(self, problem._solution)
                solver.set_parameters(problem._linear_solver_parameters)
                solver.solve()

    # return value (a class) for the decorator
    return LinearProblem_Class
