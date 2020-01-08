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

from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.parabolic.parabolic_coercive_problem import ParabolicCoerciveProblem
from rbnics.problems.parabolic.parabolic_problem import ParabolicProblem
from rbnics.reduction_methods.base import LinearTimeDependentPODGalerkinReduction
from rbnics.reduction_methods.elliptic import EllipticPODGalerkinReduction
from rbnics.reduction_methods.parabolic.parabolic_reduction_method import ParabolicReductionMethod

ParabolicPODGalerkinReduction_Base = LinearTimeDependentPODGalerkinReduction(ParabolicReductionMethod(EllipticPODGalerkinReduction))

# Base class containing the interface of a POD-Galerkin ROM
# for parabolic problems
@ReductionMethodFor(ParabolicProblem, "PODGalerkin")
@ReductionMethodFor(ParabolicCoerciveProblem, "PODGalerkin")
class ParabolicPODGalerkinReduction(ParabolicPODGalerkinReduction_Base):
    pass
