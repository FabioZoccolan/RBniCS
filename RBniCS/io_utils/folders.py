# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file folders.py
#  @brief Auxiliary folders class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

###########################     I/O     ########################### 
## @defgroup IO Input/output methods
#  @{

import os # for path and makedir
from RBniCS.io_utils.mpi import mpi_comm

class Folders(dict): # dict from string to string
    
    # Auxiliary class
    class Folder(object):
        def __init__(self, name):
            self.name = name
            
        # Returns True if it was necessary to create the folder
        # or if the folder was already created before, but it is
        # empty. Returs False otherwise.
        def create(self):
            return_value = False
            if mpi_comm.rank == 0 and os.path.exists(self.name) and len(os.listdir(self.name)) == 0: # already created, but empty
                return_value = True
            if mpi_comm.rank == 0 and not os.path.exists(self.name): # to be created
                return_value = True
                os.makedirs(self.name)
            return_value = mpi_comm.bcast(return_value, root=0)
            return return_value
            
        def __str__(self):
            return self.name
    
    def __init__(self, *args):
        dict.__init__(self, args)

    def __getitem__(self, key):
        # this will return a Folder object
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        # takes a string and initialize a Folder from it
        dict.__setitem__(self, key, Folders.Folder(val))
    
    # Returns True if it was necessary to create *at least* a folder
    # or if *at least* a folder was already created before, but it is
    # empty. Returs False otherwise.
    def create(self):
        global_return_value = False
        for key in self:
            return_value = self[key].create()
            global_return_value = global_return_value or return_value
        return global_return_value
#  @}
########################### end - I/O - end ########################### 

