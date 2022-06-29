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

from abc import ABCMeta, abstractmethod

class Distribution(object, metaclass=ABCMeta):
    @abstractmethod
    def sample(self, box, n):
        print("I am in sample of Distribution Class. This should be overridden.\n")
        raise NotImplementedError("The method sample is distribution-specific and needs to be overridden.")

#Dictionaries and lists share the following characteristics: Both are mutable, both are dynamic. They can grow and shrink as needed, both can be nested. A list can contain another list. A dictionary can contain another dictionary. A dictionary can also contain a list, and vice versa.
#Dictionaries differ from lists primarily in how elements are accessed: List elements are accessed by their position in the list, via indexing, Dictionary elements are accessed via keys.
        
    # Override the following methods to use a Distribution as a dict key
    def __hash__(self):
        dict_for_hash = list()
        for (k, v) in self.__dict__.items():
            if isinstance(v, dict): #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
                dict_for_hash.append(tuple(v.values()))
            elif isinstance(v, list): #If the type parameter is a tuple, this function will return True if the object is one of the types in the tuple.
                dict_for_hash.append(tuple(v))
            else:
                dict_for_hash.append(v)
        return hash((type(self).__name__, tuple(dict_for_hash)))

#tuples are identical to lists in all respects, except for the following properties:
#Tuples are defined by enclosing the elements in parentheses (()) instead of square brackets ([]).
#Tuples are immutable.
      
    def __eq__(self, other):
        return (type(self).__name__, self.__dict__) == (type(other).__name__, other.__dict__)
        
    def __ne__(self, other):
        return not(self == other)
