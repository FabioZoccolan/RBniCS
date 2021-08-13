# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *


@OnlineStabilization()
class AdvectionDominated(EllipticCoerciveProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V) 
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        # Store forcing expression and boundary conditions
        #self.f = Constant(1.0)
        self.bc1 = Expression("1.", element=self.V.ufl_element()) # Expression("1,0", element=self.V.ufl_element())
        self.bc2 = Expression("0.+ 1.0*(x[0] == 0.0)*(x[1] == 0.25) ", element=self.V.ufl_element()) #+ 1.0*(x[0] == 0.0)*(x[1] == 0.25) + 1.0*(x[0] == 1.0)*(x[1] == 0.0)"
         self.f = Constant(0.0)
        
        # Store terms related to stabilization
        
        self.delta = 2.1
        self.h = CellDiameter(V.mesh())

    # Return custom problem name
    def name(self):
        return "AdvectionSquareRB_N_4652_NOTSTAB_mu_2e4_0.8"
    
    # Return stability factor
    def get_stability_factor_lower_bound(self):
        return 1.

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 1.0/(mu[0])
            theta_a1 = cos(mu[1])
            theta_a2 = sin(mu[1])
            if self.stabilized:
                delta = self.delta
                theta_a3 = 0.0 #delta * cos(mu[1])**2
                theta_a4 = 0.0 #delta * cos(mu[1]) * sin(mu[1])
                theta_a5 = 0.0 #delta * sin(mu[1])**2
            else:
                theta_a3 = 0.0
                theta_a4 = 0.0
                theta_a5 = 0.0
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5)
        elif term == "f":
            theta_f0 = 0.0 
            return (theta_f0,)
        elif term == "dirichlet_bc":
            theta_bc0 = 1.0
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")


    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        ds = self.ds  
        if term == "a":
            u = self.u
            h = self.h
            a0 = inner(grad(u), grad(v)) * dx 
            a1 = u.dx(0) * v * dx  
            a2 = u.dx(1) * v * dx
            a3 = h * u.dx(0) * v.dx(0) * dx 
            a4 = h * u.dx(0) * v.dx(1) * dx + h * u.dx(1) * v.dx(0) * dx 
            a5 = h * u.dx(1) * v.dx(1) * dx 
            return (a0, a1, a2, a3, a4, a5)
         elif term == "f":
            f = self.f
            f0 = f*v dx
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, self.bc1, self.boundaries, 1),
                   DirichletBC(self.V, self.bc2, self.boundaries, 2)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/square_N_4652.xml")
subdomains = MeshFunction("size_t", mesh, "data/square_N_4652_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_N_4652_facet_region.xml")
print("hMax: ", mesh.hmax() )

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)
print("Dim: ", V.dim() )

# 3. Allocate an object of the AdvectionDominated class
problem = AdvectionDominated(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1e4, 5e5), (0.0, 6.3)]
problem.set_mu_range(mu_range)


### Offline Solution
offline_mu = (2e4, 0.8)
problem.set_mu(offline_mu)
problem.init()

problem.solve()
problem.export_solution(filename="FEM_adsquare_NOSTAB_N_4652_mu_2e4_0.8") #_with_stabilization



# 4. Prepare reduction with a reduced basis method
reduction_method = ReducedBasis(problem)
reduction_method.set_Nmax(50)
reduction_method.set_tolerance(1e-7)

# 5. Perform the offline phase
lifting_mu = (2e4, 0.8)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(200)
reduced_problem = reduction_method.offline()


# 6. Perform an online solve
online_mu = (2e4, 0.8)
reduced_problem.set_mu(online_mu)

reduced_problem.solve(online_stabilization=True)
reduced_problem.export_solution(filename="online_solution_adsquare_NOSTAB_N_4652_with_stabilization_mu_2e4_0.8") 

reduced_problem.solve(online_stabilization=False)
reduced_problem.export_solution(filename="online_solution_adsquare_NOSTAB_N_4652_without_stabilization_mu_2e4_0.8")
# 7. Perform an error analysis
reduction_method.initialize_testing_set(100)
reduction_method.error_analysis(online_stabilization=True, filename="error_analysis_adsquare_NOSTAB_N_4652_with_stabilization_mu_2e4_0.8")
reduction_method.error_analysis(online_stabilization=False, filename="error_analysis_adsquare_NOSTAB_N_4652_without_stabilization_mu_2e4_0.8")

# 8. Perform a speedup analysis
reduction_method.speedup_analysis(online_stabilization=True, filename="speedup_analysis_adsquare_NOSTAB_N_4652_with_stabilization_mu_2e4_0.8")
reduction_method.speedup_analysis(online_stabilization=False, filename="speedup_analysis_adsquare_NOSTAB_N_4652_without_stabilization_mu_2e4_0.8")
