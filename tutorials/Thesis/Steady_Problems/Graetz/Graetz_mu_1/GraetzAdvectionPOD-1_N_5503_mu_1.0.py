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
        # Store advection and forcing expressions
        self.vel = Expression("x[1]*(1-x[1])", element=self.V.ufl_element())
        self.f = Constant(0.0)
        self.lifting = Expression('((x[0] >= 1 && x[0] <= 2) && (x[1] == 1.0 || x[1]== 0.0) ) ? 1. : 0.', degree=1, domain=mesh)
        # Store terms related to stabilization
        self.delta = 1.0
        self.h = CellDiameter(V.mesh())

    # Return custom problem name
    def name(self):
        return "AdvectionPODGraetz-1_N_5503_STAB_mu_1.0_d_1"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
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
        elif term == "f":
            theta_f0 = 0.0 #- 1/(mu[0])
            theta_f1 = 0.0 #-4.0
            if self.stabilized:
                delta = self.delta
                theta_f2 = 0.0 #-delta * 4.0
                theta_f3 = 0.0 #-delta * 4.0
            else:
                theta_f2 = 0.0
                theta_f3 = 0.0
            return (theta_f0, theta_f1, theta_f2, theta_f3)
        elif term == "dirichlet_bc":
            theta_bc0 = 1.0
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            vel = self.vel
            h = self.h
            a0 = inner(grad(u), grad(v)) * dx #(1)
            a1 = vel * u.dx(0) * v * dx #(1)
            a2 = h * vel * u.dx(0) * v.dx(0) * dx(1) #in case, take all the domain
            a3 = h * vel * u.dx(0) * v.dx(0) * dx(2)
            return (a0, a1, a2, a3)
        elif term == "f":
            l = self.lifting
            vel = self.vel
            h = self.h
            f0 = inner(grad(l), grad(v)) * dx
            f1 = h * vel * l.dx(0) * v * dx
            f2 = h * vel * l.dx(0) * v.dx(0) * dx(1)
            f3 = h * vel * l.dx(0) * v.dx(0) * dx(2) 
            return (f0, f1, f2, f3)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 2),
                   DirichletBC(self.V, Constant(1.0), self.boundaries, 4),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 5),
                   DirichletBC(self.V, Constant(0.0), self.boundaries, 6)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/graetz_N_5503.xml")
print("hmax", mesh.hmax())
subdomains = MeshFunction("size_t", mesh, "data/graetz_N_5503_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetz_N_5503_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)
print("FE dim", V.dim())

# 3. Allocate an object of the AdvectionDominated class
problem = AdvectionDominated(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.01, 1e6), (0.5,4.0)]
problem.init()
problem.set_mu_range(mu_range)
offline_mu = (1.0, 1.0)

problem.set_mu(offline_mu)
problem.solve()
problem.export_solution(filename="FEM_Graetz_STAB_N_5503_offline_mu_1.0_d_1")


# 4. Prepare reduction with a reduced basis method
reduction_method = PODGalerkin(advection_dominated_problem)
reduction_method.set_Nmax(50)
reduction_method.set_tolerance(1e-7)

# 5. Perform the offline phase
lifting_mu = (1.0, 1.0)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(200)
reduced_problem = reduction_method.offline()

# 6. Perform an online solve
online_mu = (1.0, 1.0)
reduced_problem.set_mu(online_mu)
reduced_problem.solve(online_stabilization=True)
reduced_problem.export_solution(filename="online_solution_Graetz_STAB_N_5503_with_stabilization_mu_1.0_d_1")
reduced_problem.solve(online_stabilization=False)
reduced_problem.export_solution(filename="online_solution_Graetz_STAB_N_5503_without_stabilization_mu_1.0_d_1")

# 7. Perform an error analysis
reduction_method.initialize_testing_set(100)
reduction_method.error_analysis(online_stabilization=True, filename="error_analysis_Graetz_STAB_N_5503_with_stabilization_mu_1.0_d_1")
reduction_method.error_analysis(online_stabilization=False, filename="error_analysis_Graetz_STAB_N_5503_without_stabilization_mu_1.0_d_1")

# 8. Perform a speedup analysis
reduction_method.speedup_analysis(online_stabilization=True, filename="speedup_analysis_Graetz_STAB_N_5503_with_stabilization_mu_1.0_d_1")
reduction_method.speedup_analysis(online_stabilization=False, filename="speedup_analysis_Graetz_STAB_N_5503_without_stabilization_mu_1.0_d_1") 
