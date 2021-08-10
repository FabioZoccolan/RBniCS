# Copyright (C) 2015-2019 by the RBniCS authors
# Copyright (C) 2016-2019 by the multiphenics authors
#
# This file is part of the RBniCS interface to multiphenics.
#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from multiphenics import *
from rbnics import *
from problems import *
from reduction_methods import *

class EllipticOptimalControl(ParabolicCoerciveProblem): #ParabolicCoerciveProblem #EllipticOptimalControlProblem
    
    # Default initialization of members
    def __init__(self, block_V, **kwargs):
        # Call the standard initialization
        ParabolicCoerciveProblem.__init__(self, block_V, **kwargs) #ParabolicCoerciveProblem #EllipticOptimalControlProblem
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        block_yup = BlockTrialFunction(block_V)
        (self.y, self.u, self.p) = block_split(block_yup)
        block_zvq = BlockTestFunction(block_V)
        (self.z, self.v, self.q) = block_split(block_zvq)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Regularization coefficient
        self.alpha = 0.01
        self.y_d = Constant(1.0)
        # Store the velocity expression
        self.vel = Expression("x[1]*(1-x[1])", element=self.V.sub(0).ufl_element())
        # Store the initial condition expression
        self.ic = Expression("1.0", element=self.V.sub(0).ufl_element())
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "mumps"
        })
        
    # Return custom problem name
    def name(self):
        return "ParabolicOCGraetzRB_N_18798_mu_1e5_alpha_0.01"
        
    # Return stability factor
    def get_stability_factor_lower_bound(self):
        return min(self.compute_theta("a"))
    
    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "p":
            theta_p0 = 1.
            return (theta_p0, )
        if term in ("a", "a*"):
            theta_a0 = 1.0/mu[0]
            theta_a1 = 4.0
            return (theta_a0, theta_a1)
        elif term in ("c", "c*"):
            theta_c0 = 1.0
            return (theta_c0,)
        elif term == "m":
            theta_m0 = 1.0
            return (theta_m0,)
        elif term == "n":
            theta_n0 = self.alpha
            return (theta_n0,)
        elif term == "f":
            theta_f0 = 0.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.0
            return (theta_g0,)
        elif term == "h":
            theta_h0 = 1**2
            return (theta_h0,)
        elif term == "dirichlet_bc_y":
            theta_bc0 = 1.
            return (theta_bc0,)
        elif term == "initial_condition":
            theta_ic0 = 1.0
            return (theta_ic0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
    
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        print(term)
        dx = self.dx
        if term == "p":
            y = self.y
            q = self.q
            p0 = [[0, 0, 0], [0, 0, 0], [y*q*dx, 0, 0]]
            return (p0)
        if term == "a":
            y = self.y
            q = self.q
            vel = self.vel
            a0 = [[0, 0, 0], [0, 0, 0], [inner(grad(y), grad(q))*dx, 0, 0]]
            a1 = [[0, 0, 0], [0, 0, 0], [vel*y.dx(0)*q*dx, 0, 0]]
            return (a0, a1)
        elif term == "a*":
            z = self.z
            p = self.p
            vel = self.vel
            as0 = [[0, 0, inner(grad(z), grad(p))*dx], [0, 0, 0], [0, 0, 0]]
            as1 = [[0, 0, - vel*p.dx(0)*z*dx], [0, 0, 0], [0, 0, 0]]
            return (as0, as1)
        elif term == "c":
            u = self.u
            q = self.q
            c0 = [[0, 0, 0], [0, 0, 0], [0, u*q*dx, 0]]
            return (c0,)
        elif term == "c*":
            v = self.v
            p = self.p
            cs0 = [[0, 0, 0], [0, 0, v*p*dx], [0, 0, 0]]
            return (cs0,)
        elif term == "m":
            y = self.y
            z = self.z
            m0 = [[y*z*dx, 0, 0], [0, 0, 0], [0, 0, 0]] #y*z*dx(1) + y*z*dx(2)
            return (m0,)
        elif term == "n":
            u = self.u
            v = self.v
            n0 = [[0, 0, 0], [0, u*v*dx, 0], [0, 0, 0]]
            return (n0,)
        elif term == "f":
            q = self.q
            f0 = [0, 0, Constant(0.0)*q*dx]
            return (f0,)
        elif term == "g":
            y_d = self.y_d
            z = self.z
            g0 = [y_d * z * dx(3) + y_d * z * dx(4), 0, 0]
            return (g0)
        elif term == "h":
            h0 = 1.0
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = BlockDirichletBC([[DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V.sub(0), Constant(1.0), self.boundaries, 2),
                   DirichletBC(self.V.sub(0), Constant(1.0), self.boundaries, 4),
                   DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 5),
                   DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 6)], None, None])
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = BlockDirichletBC([None, None, [DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 2),
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 4),
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 5),
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 6)]])
            return (bc0,)
        elif term == "inner_product_y":
            y = self.y
            z = self.z
            x0 = [[inner(grad(y), grad(z))*dx, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (x0,)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0 = [[0, 0, 0], [0, u*v*dx, 0], [0, 0, 0]]
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = [[0, 0, 0], [0, 0, 0], [0, 0, inner(grad(p), grad(q))*dx]]
            return (x0,)
        elif term == "projection_inner_product":
            u = self.u
            x0 = [[u*v*dx], 0, 0]
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

# 1. Read the mesh for this problem
mesh = Mesh("data/graetzOC_N_18798.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetzOC_N_18798_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetzOC_N_18798_facet_region.xml")
print("hMax: ", mesh.hmax() )

# 2. Create Finite Element space (Lagrange P1)
scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = BlockElement(scalar_element, scalar_element, scalar_element)
block_V = BlockFunctionSpace(mesh, element, components=["y", "u", "p"])
print("Dim: ", block_V.dim() )

# 3. Allocate an object of the EllipticOptimalControl class
elliptic_optimal_control = EllipticOptimalControl(block_V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.01, 1e6), (0.5, 4.0)]
elliptic_optimal_control.set_mu_range(mu_range)
elliptic_optimal_control.set_time_step_size(0.05)
elliptic_optimal_control.set_final_time(3)

offline_mu = (1e5, 1.2)
elliptic_optimal_control.init()
elliptic_optimal_control.set_mu(offline_mu)
elliptic_optimal_control.solve()
elliptic_optimal_control.export_solution(filename="FEM_OCGraetz_N_18798_mu_1e5_alpha_0.01")

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(elliptic_optimal_control)
reduced_basis_method.set_Nmax(40)

# 5. Perform the offline phase
lifting_mu = (1e5, 1.2)
elliptic_optimal_control.set_mu(lifting_mu)
reduced_basis_method.initialize_training_set(100)
reduced_elliptic_optimal_control = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (1e5, 1.2)
reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_elliptic_optimal_control.solve()
reduced_elliptic_optimal_control.export_solution(filename="online_solution")
print("Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(100)
reduced_basis_method.error_analysis()

# 8. Perform a speedup analysis
reduced_basis_method.speedup_analysis()
