# Copyright (C) 2016-2021 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from mshr import *
from multiphenics import *

"""
This file generates the mesh which is used in the following examples:
    3b_advection_diffusion_reaction_neumann_control
The test case is from section 5.3 of
F. Negri, G. Rozza, A. Manzoni and A. Quarteroni. Reduced Basis Method for Parametrized Elliptic Optimal Control Problems. SIAM Journal on Scientific Computing, 35(5): A2316-A2340, 2013.
"""

# Create mesh
domain = Rectangle(Point(0., 0.), Point(2., 1.))
#subdomain = dict()
subdomain1 = Rectangle(Point(0., 0.), Point(1., 1.))
subdomain2 = Rectangle(Point(1., 0.2), Point(2., 0.8))
subdomain3 = Rectangle(Point(1., 0.), Point(2., 0.2))
subdomain4 = Rectangle(Point(1., 0.8), Point(2., 1.))
domain.set_subdomain(1, subdomain1)  
domain.set_subdomain(2, subdomain2) 
domain.set_subdomain(3, subdomain3) 
domain.set_subdomain(4, subdomain4) 
mesh = generate_mesh(domain, 70)
plot(mesh)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
plot(subdomains)

# Create boundaries
class Left(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS


class Right(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return on_boundary and ( abs(x[0] - 2.) < DOLFIN_EPS and ( x[1] <= 0.8 and x[1] >=0.2) )
        

class Up_Down_Control2(SubDomain):
    def __init__(self, x_min, x_max ):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max

    def inside(self, x, on_boundary):
        return on_boundary and  (( (abs(x[1] - 1.) < DOLFIN_EPS and (x[0] >= self.x_min and x[0] <= self.x_max) ) or ( abs(x[1] - 0.) < DOLFIN_EPS and (x[0] >= self.x_min and x[0] <= self.x_max ) )  ) or  (abs(x[0] - 2.) < DOLFIN_EPS and ( x[1] >= 0.8 or x[1] <= 0.2)) )
        
        
class Bottom(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max


class Top(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

bottomLeft = Bottom(0., 1.)
bottomLeft.mark(boundaries, 1)

up_down_control = Up_Down_Control2(1., 2.)
up_down_control.mark(boundaries, 2)


right = Right()
right.mark(boundaries, 3)


topLeft = Top(0., 1.)
topLeft.mark(boundaries, 4)

left = Left()
left.mark(boundaries, 5)

# Create restrictions
control_restriction = MeshRestriction(mesh, up_down_control)

# Save
File("graetz_BC2.xml") << mesh
File("graetz_BC2_physical_region.xml") << subdomains
File("graetz_BC2_facet_region.xml") << boundaries
File("graetz_BC2_restriction_control.rtc.xml") << control_restriction
XDMFFile("graetz_BC2.xdmf").write(mesh)
XDMFFile("graetz_BC2_physical_region.xdmf").write(subdomains)
XDMFFile("graetz_BC2_facet_region.xdmf").write(boundaries)
XDMFFile("graetz_BC2_restriction_control.rtc.xdmf").write(control_restriction)
