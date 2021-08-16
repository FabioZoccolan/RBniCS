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

from numpy import isclose
from dolfin import *
import matplotlib.pyplot as plt
from multiphenics import *

r"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Gamma_2} (y - y_d)^2 ds + \alpha/2 \int_{\Gamma_2} u^2 ds
s.t.
      - \Delta y = f       in \Omega
    \partial_n y = 0       on \Gamma_1
    \partial_n y = u       on \Gamma_2
    \partial_n y = 0       on \Gamma_3
               y = 0       on \Gamma_4

where
    \Omega                      unit square
    u \in L^2(\Gamma_2)         control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d = 1                     desired state
    f                           forcing term

using an adjoint formulation solved by a one shot approach.
Note that this is an alternative, to Lagrange multipliers, weak imposition
of y = y_d on \Gamma_2.
"""

# MESH #
# Mesh
mesh = Mesh("data/graetz_BC.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetz_BC_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetz_BC_facet_region.xml")
print("hMax: ", mesh.hmax() )
# Dirichlet boundary
right = MeshRestriction(mesh, "data/graetz_BC_restriction_control.rtc.xml")

# FUNCTION SPACES #
Y = FunctionSpace(mesh, "Lagrange", 1)
U = FunctionSpace(mesh, "Lagrange", 1)
Q = Y
W = BlockFunctionSpace([Y, U, Q], restrict=[None, right, None])
print("Dim: ", W.dim() )

# PROBLEM DATA #
alpha = Constant(0.01)
y_d = Constant(1.)
vel = Expression("x[1] * (1 - x[1])", element=W.sub(0).ufl_element())
f = Constant(0.0) #Expression("10*sin(2*pi*x[0])*sin(2*pi*x[1])", element=W.sub(0).ufl_element())
mu1 = 1e5


# TRIAL/TEST FUNCTIONS #
yup = BlockTrialFunction(W)
(y, u, p) = block_split(yup)
zvq = BlockTestFunction(W)
(z, v, q) = block_split(zvq)

# MEASURES #
ds = Measure("ds")(subdomain_data=boundaries)

# OPTIMALITY CONDITIONS #
a = [[y*z*dx(3)+y*z*dx(4)               , 0              , -1/mu1 * inner(grad(p), grad(z))*dx + 4.0 * vel * p.dx(0) * z * dx],
     [0                         , alpha*u*v*ds(0),  p*v*ds(0)               ],
     [-1/mu1 * inner(grad(y), grad(q))*dx - 4.0 * vel * y.dx(0) * q * dx, u*q*ds(0)    , 0                         ]]
f =  [y_d*z*dx(3)+y_d*z*dx(4) ,
      0          ,
      f*q*dx      ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), Constant(1.0), boundaries, 1),
                   DirichletBC(W.sub(0), Constant(1.0), boundaries, 2),
                   DirichletBC(W.sub(0), Constant(1.0), boundaries, 4),
                   DirichletBC(W.sub(0), Constant(1.0), boundaries, 5),
                   DirichletBC(W.sub(0), Constant(1.0), boundaries, 6)],
                       [],
                       [DirichletBC(W.sub(2), Constant(0.0), boundaries, 1),
                   DirichletBC(W.sub(2), Constant(0.0), boundaries, 2),
                   DirichletBC(W.sub(2), Constant(0.0), boundaries, 4),
                   DirichletBC(W.sub(2), Constant(0.0), boundaries, 5),
                   DirichletBC(W.sub(2), Constant(0.0), boundaries, 6)]])

# SOLUTION #
yup = BlockFunction(W)
(y, u, p) = block_split(yup)

# FUNCTIONAL #
J = 0.5*inner(y - y_d, y - y_d)*dx(3) + 0.5*inner(y - y_d, y - y_d)*dx(4) + 0.5*alpha*inner(u, u)*ds(2)

# UNCONTROLLED FUNCTIONAL VALUE #
A_state = assemble(a[2][0])
F_state = assemble(f[2])
[bc_state.apply(A_state) for bc_state in bc[0]]
[bc_state.apply(F_state) for bc_state in bc[0]]
solve(A_state, y.vector(), F_state)
print("Uncontrolled J =", assemble(J))
#assert isclose(assemble(J), 0.5295414)
plt.figure()
plot(y, title="uncontrolled state")
plt.show()

# OPTIMAL CONTROL #
A = block_assemble(a, keep_diagonal=True)
F = block_assemble(f)
bc.apply(A)
bc.apply(F)
block_solve(A, yup.block_vector(), F)
print("Optimal J =", assemble(J))
#assert isclose(assemble(J), 8.0770264e-6)
plt.figure()
plot(y, title="state")
plt.figure()
plot(u, title="control")
plt.figure()
plot(p, title="adjoint")
plt.show()
