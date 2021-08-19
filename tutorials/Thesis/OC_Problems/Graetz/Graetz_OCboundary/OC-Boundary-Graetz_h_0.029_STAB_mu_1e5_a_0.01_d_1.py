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
from multiphenics import *
from rbnics import *
from problems import *
from reduction_methods import *
from numpy import ceil, zeros, isclose 

"""### 3. Affine Decomposition

For this problem the affine decomposition is straightforward.
"""

@OnlineStabilization()
class EllipticOptimalControl(EllipticOptimalControlProblem):

    # Default initialization of members
    def __init__(self, block_V, **kwargs):
        # Call the standard initialization
        EllipticOptimalControlProblem.__init__(self, block_V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        block_yup = BlockTrialFunction(block_V)
        (self.y, self.u, self.p) = block_split(block_yup)
        block_zvq = BlockTestFunction(block_V)
        (self.z, self.v, self.q) = block_split(block_zvq)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        
        # Regularization coefficient
        self.alpha = 0.0001
        self.y_d = Constant(1.0)
        
        # Store the velocity expression
        self.vel = Expression("x[1] * (1 - x[1])", degree=1, domain=mesh)
  
        self.f_0 = Constant(1.0)
        
        self.delta = 1.0
        self.h = CellDiameter(block_V.mesh())
        
        
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "umfpack"
        })

    # Return custom problem name
    def name(self):
        return "Boundary_OCGraetzPOD_h_0.029_mu_1e5_alpha_0.01"
    
    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term in ("a", "a*"):
            theta_a0 = 1/(mu[0])
            theta_a1 = 4.0
            if self.stabilized:
              delta = self.delta
              theta_a2 = delta * 4.0
              theta_a3 = delta * 4.0
            else:
              theta_a2 = 0.0
              theta_a3 = 0.0
            return (theta_a0, theta_a1, theta_a2, theta_a3)
        elif term in ("c", "c*"):
            theta_c0 = 1.0
            if self.stabilized:
               delta = self.delta
               theta_c1 = delta
            else:
               theta_c1 = 0.0
            return (theta_c0,theta_c1)
        elif term == "m":
            theta_m0 = 1.0
            if self.stabilized:
                delta = self.delta
                theta_m1 = delta
            else:
                theta_m1 = 0.0
            return (theta_m0, theta_m1)
        elif term == "n":
            theta_n0 = self.alpha
            return (theta_n0,)
        elif term == "f":
            theta_f0 = 0.0
            return (theta_f0, )
        elif term == "g":
            theta_g0 = 1.
            if self.stabilized:
               delta = self.delta
               theta_g1 = delta
            else:
               theta_g1 = 0.0
            return (theta_g0, theta_g1)
        elif term == "h":
            theta_h0 = 1.
            return (theta_h0,)
        elif term == "dirichlet_bc_y":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    def assemble_operator(self, term):
        print(term)
        dx = self.dx
        ds = self.ds
        if term == "a":
            y = self.y
            q = self.q
            vel = self.vel  
            h = self.h
            a0_0 = inner(grad(y), grad(q))*dx
            a1_0 = vel * y.dx(0) * q * dx 
            a2_0 = h * vel * y.dx(0) * q.dx(0) * dx(1)
            a3_0 = h * vel * y.dx(0) * q.dx(0) * dx(2)+ h * vel * y.dx(0) * q.dx(0) * dx(3)+h * vel * y.dx(0) * q.dx(0) * dx(4)
            a0 = [[0, 0, 0], [0, 0, 0], [a0_0, 0, 0]]
            a1 = [[0, 0, 0], [0, 0, 0], [a1_0, 0, 0]]
            a2 = [[0, 0, 0], [0, 0, 0], [a2_0, 0, 0]]
            a3 = [[0, 0, 0], [0, 0, 0], [a3_0, 0, 0]]
            return (BlockForm(a0), BlockForm(a1),BlockForm(a2), BlockForm(a3))
        elif term == "a*":
            z = self.z
            p = self.p
            vel = self.vel
            h = self.h
            as0_0 = inner(grad(p), grad(z))*dx
            as1_0 = -vel * p.dx(0) * z * dx 
            as2_0 = h * vel * p.dx(0) * z.dx(0) * dx(1)
            as3_0 = h * vel * p.dx(0) * z.dx(0) * dx(2) + h * vel * p.dx(0) * z.dx(0) * dx(3) + h * vel * p.dx(0) * z.dx(0) * dx(4)
            as0 = [[0, 0, as0_0], [0, 0, 0], [0, 0, 0]]
            as1 = [[0, 0, as1_0], [0, 0, 0], [0, 0, 0]]
            as2 = [[0, 0, as2_0], [0, 0, 0], [0, 0, 0]]
            as3 = [[0, 0, as3_0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(as0), BlockForm(as1), BlockForm(as2), BlockForm(as3))
        elif term == "c":
            u = self.u
            q = self.q
            h = self.h
            c0_0 = u*q*ds(7)
            c1_0 = h * u * q.dx(0) * ds(7)
            c0 = [[0, 0, 0], [0, 0, 0], [0, c0_0, 0]]
            c1 = [[0, 0, 0], [0, 0, 0], [0, c1_0, 0]]
            return(BlockForm(c0),BlockForm(c1))
        elif term == "c*":
            v = self.v
            p = self.p
            h = self.h
            cs0_0 = p*v*ds(7)
            cs1_0 = Constant(0.0) * - h * v.dx(0) * p * ds(7)
            cs0 = [[0, 0, 0], [0, 0, cs0_0], [0, 0, 0]]
            cs1 = [[0, 0, 0], [0, 0, cs1_0], [0, 0, 0]]
            return(BlockForm(cs0),BlockForm(cs1))
        elif term == "m":
            y = self.y
            z = self.z
            h = self.h
            m0_0 =  y * z *dx(3) + y * z *dx(4)
            m1_0 = -h * y * z.dx(0) * dx(4)-h * y * z.dx(0) * dx(3)
            m0 = [[m0_0, 0, 0], [0, 0, 0], [0, 0, 0]]
            m1 = [[m1_0, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(m0),BlockForm(m1))
        elif term == "n":
            u = self.u
            v = self.v
            h = self.h
            n0_0 = u*v*ds(7)
            n0 = [[0, 0, 0], [0, n0_0, 0], [0, 0, 0]]
            return (BlockForm(n0),)
        elif term == "f":
            q = self.q
            f_0 = self.f_0
            f0_0 = f_0 * q *dx 
            f0 = [0, 0, f0_0] 
            return (BlockForm(f0),)
        elif term == "g":
            y_d = self.y_d
            z = self.z
            h = self.h
            g0_0 = y_d * z * dx(3) + y_d * z* dx(4)
            g1_0 = - h * y_d * z.dx(0) * dx(3) - h * y_d * z.dx(0) * dx(4)
            g0 = [g0_0, 0, 0]
            g1 = [g1_0, 0, 0]
            return (BlockForm(g0),BlockForm(g1))
        elif term == "h":
            y_d = self.y_d
            h0 = y_d * y_d * dx(3, domain=mesh) + y_d * y_d * dx(4, domain=mesh)  #RICONTROLLARE
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = BlockDirichletBC([[DirichletBC(block_V.sub(0), Constant(0.0), self.boundaries, 1),
                   DirichletBC(block_V.sub(0), Constant(1.0), self.boundaries, 2),
                   DirichletBC(block_V.sub(0), Constant(1.0), self.boundaries, 4),
                   DirichletBC(block_V.sub(0), Constant(0.0), self.boundaries, 5),
                   DirichletBC(block_V.sub(0), Constant(0.0), self.boundaries, 6)], None, None])
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = BlockDirichletBC([None, None, [DirichletBC(block_V.sub(2), Constant(0.0), self.boundaries, 1),
                   DirichletBC(block_V.sub(2), Constant(0.0), self.boundaries, 2),
                   DirichletBC(block_V.sub(2), Constant(0.0), self.boundaries, 4),
                   DirichletBC(block_V.sub(2), Constant(0.0), self.boundaries, 5),
                   DirichletBC(block_V.sub(2), Constant(0.0), self.boundaries, 6)]] )
            return (bc0,)
        elif term == "inner_product_y":
            y = self.y
            z = self.z
            x0_y = inner(grad(y), grad(z))*dx
            x0 = [[x0_y, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(x0),)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0_u = u* v *ds(7)
            x0 = [[0, 0, 0], [0, x0_u, 0], [0, 0, 0]]
            return (BlockForm(x0),)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0_p = inner(grad(p), grad(q))*dx
            x0 = [[0, 0, 0], [0, 0, 0], [0, 0, x0_p]]
            return (BlockForm(x0),)
        else:
            raise ValueError("Invalid term for assemble_operator().")
            
            
# MESH #
# Mesh
mesh = Mesh("data/graetz_BC.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetz_BC_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetz_BC_facet_region.xml")
print("hMax: ", mesh.hmax() )
# Dirichlet boundary
control_boundary = MeshRestriction(mesh, "data/graetz_BC_restriction_control.rtc.xml")

#BOUNDARY RESTRICTIONS #
y_restrict = []
u_restrict = []
p_restrict = []

y_restrict.append(None)
u_restrict.append(control_boundary)
p_restrict.append(None)
   
# FUNCTION SPACES #
scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = BlockElement([scalar_element] + [scalar_element] + [scalar_element])
components = ["y"]+ ["u"] + ["p"]
block_V = BlockFunctionSpace(mesh, element,  restrict = [*y_restrict, *u_restrict, *p_restrict] , components=[*components])
print("Dim: ", block_V.dim() )


# 3. Allocate an object of the EllipticOptimalControl class
elliptic_optimal_control = EllipticOptimalControl(block_V, subdomains=subdomains, boundaries=boundaries)
mu_range =  [(0.01, 1e6), (0.5, 4.0)]
elliptic_optimal_control.set_mu_range(mu_range)

offline_mu = (1e5, 1.0)
elliptic_optimal_control.init()
elliptic_optimal_control.set_mu(offline_mu)
elliptic_optimal_control.solve()
elliptic_optimal_control.export_solution(filename="BoundaryFEM_OCGraetz_h_0.029_mu_1e5_alpha_0.01")



# ### 4.4. Prepare reduction with a reduced basis method

# In[ ]:

pod_galerkin_method = PODGalerkin(elliptic_optimal_control)
pod_galerkin_method.set_Nmax(3)

# ### 4.5. Perform the offline phase

# In[ ]:


lifting_mu = (1e5, 1.0)
elliptic_optimal_control.set_mu(lifting_mu)
pod_galerkin_method.initialize_training_set(5)
reduced_elliptic_optimal_control = pod_galerkin_method.offline()


# ### 4.6. Perform an online solve

# In[ ]:


online_mu = (1e5, 1.0)
reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_solution = reduced_elliptic_optimal_control.solve()
print("Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())
reduced_elliptic_optimal_control.export_solution(filename="Boundary_online_solution_OCGraetz_h_0.029_mu_1e5_alpha_0.01")

# In[ ]:




# ### 4.7. Perform an error analysis

# In[ ]:

pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()

