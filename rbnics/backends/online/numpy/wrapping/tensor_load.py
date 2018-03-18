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

from rbnics.utils.io import NumpyIO

def tensor_load(tensor, directory, filename):
    if NumpyIO.exists_file(directory, filename):
        loaded = NumpyIO.load_file(directory, filename)
        assert len(loaded.shape) in (1, 2)
        if len(loaded.shape) is 1:
            tensor[:] = loaded
        elif len(loaded.shape) is 2:
            tensor[:, :] = loaded
        else:
            raise ValueError("Invalid tensor shape")
        return True
    else:
        return False
