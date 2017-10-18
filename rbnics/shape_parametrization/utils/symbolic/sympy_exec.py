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

import math
from numbers import Number
import sympy

def sympy_exec(string, locals):
    locals = dict(locals)
    locals.update(math_sympy_locals)
    exec(string, {"__builtins__": None}, locals) # stores the result in an expression named e
    return e  # noqa
    
math_sympy_locals = dict()
for package in (math, sympy):
    for name, item in package.__dict__.items():
        if callable(item) or isinstance(item, Number):
            math_sympy_locals[name] = item
