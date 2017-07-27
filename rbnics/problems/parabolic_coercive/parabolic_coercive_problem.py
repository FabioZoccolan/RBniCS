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

from rbnics.problems.base import LinearTimeDependentProblem
from rbnics.problems.elliptic_coercive import EllipticCoerciveProblem
from rbnics.backends import product, sum
from rbnics.utils.decorators import Extends, override

ParabolicCoerciveProblem_Base = LinearTimeDependentProblem(EllipticCoerciveProblem)

# Base class containing the definition of parabolic coercive problems
@Extends(ParabolicCoerciveProblem_Base)
class ParabolicCoerciveProblem(ParabolicCoerciveProblem_Base):
    
    ## Default initialization of members
    @override
    def __init__(self, V, **kwargs):
        # Call to parent
        ParabolicCoerciveProblem_Base.__init__(self, V, **kwargs)
        
        # Form names for parabolic problems
        self.terms.append("m")
        self.terms_order.update({"m": 2})
        
    class ProblemSolver(ParabolicCoerciveProblem_Base.ProblemSolver):
        def residual_eval(self, t, solution, solution_dot):
            problem = self.problem
            problem.set_time(t)
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"]))
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            assembled_operator["f"] = sum(product(problem.compute_theta("f"), problem.operator["f"]))
            return (
                  assembled_operator["m"]*solution_dot
                + assembled_operator["a"]*solution
                - assembled_operator["f"]
            )
            
        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            problem = self.problem
            problem.set_time(t)
            assembled_operator = dict()
            assembled_operator["m"] = sum(product(problem.compute_theta("m"), problem.operator["m"]))
            assembled_operator["a"] = sum(product(problem.compute_theta("a"), problem.operator["a"]))
            return (
                  assembled_operator["m"]*solution_dot_coefficient
                + assembled_operator["a"]
            )
                    
