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

from rbnics.backends.abstract import SymbolicParameters as AbstractSymbolicParameters
from rbnics.utils.decorators import BackendFor, tuple_of

# Handle the trivial case of a non-parametric problem, that is mu = ()
@BackendFor("common", inputs=(object, object, tuple_of(())))
class SymbolicParameters(AbstractSymbolicParameters, tuple):
    def __new__(cls, problem, V, mu):
        return tuple.__new__(cls, ())
        
    def __str__(self):
        assert len(self) == 0
        return "()"
