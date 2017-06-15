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

from rbnics.reduction_methods.base.linear_reduction_method import LinearReductionMethod
from rbnics.reduction_methods.base.time_dependent_reduction_method import TimeDependentReductionMethod
from rbnics.utils.decorators import apply_decorator_only_once, Extends

@apply_decorator_only_once
def LinearTimeDependentReductionMethod(DifferentialProblemReductionMethod_DerivedClass):
    
    LinearTimeDependentReductionMethod_Base = TimeDependentReductionMethod(LinearReductionMethod(DifferentialProblemReductionMethod_DerivedClass))
    
    @Extends(LinearTimeDependentReductionMethod_Base, preserve_class_name=True)
    class LinearTimeDependentReductionMethod_Class(LinearTimeDependentReductionMethod_Base):
        pass
                
    # return value (a class) for the decorator
    return LinearTimeDependentReductionMethod_Class
    
