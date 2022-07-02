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
from rbnics.reduction_methods.base import NonlinearPODGalerkinReduction
from rbnics.problems.navier_stokes.navier_stokes_problem import NavierStokesProblem
from rbnics.reduction_methods.stokes import StokesPODGalerkinReduction
from rbnics.reduction_methods.navier_stokes.navier_stokes_reduction_method import NavierStokesReductionMethod

NavierStokesPODGalerkinReduction_Base = NonlinearPODGalerkinReduction(NavierStokesReductionMethod(StokesPODGalerkinReduction))

@ReductionMethodFor(NavierStokesProblem, "PODGalerkin")
class NavierStokesPODGalerkinReduction(NavierStokesPODGalerkinReduction_Base):
    pass