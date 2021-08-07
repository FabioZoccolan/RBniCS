from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *


# ### Affine Decomposition



#@OnlineStabilization()
class EllipticOptimalControl(EllipticOptimalControlProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticOptimalControlProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        yup = TrialFunction(V)
        (self.y, self.u, self.p) = split(yup)
        zvq = TestFunction(V)
        (self.z, self.v, self.q) = split(zvq)
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        # Regularization coefficient
        self.alpha = 0.01
        # Desired state
        self.y_d = Constant(0.5)
        
        self.delta = 2.1
        
        self.h = CellDiameter(V.mesh())
        self.bc1 = Constant(1.0)
        self.bc2 = Expression("0.0 + 1.0*(x[0] == 0.0)*(x[1] == 0.25)", element=self.V.ufl_element())
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "mumps"
        })

    # Return custom problem name
    def name(self):
        return "AdvectionOCSquareRB-1_N_3594_mu_2.4_1.2_d_2.1_alpha_0.01_STSTAD"

    # Return stability factor
    def get_stability_factor_lower_bound(self):
        return 1.

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        delta = self.delta
        if term in ("a", "a*"):
            theta_a0 = 1/(mu[0])
            theta_a1 = cos(mu[1])
            theta_a2 = sin(mu[1])
            #if self.stabilized:
            delta = self.delta
            theta_a3 = delta * cos(mu[1])**2
            theta_a4 = delta * cos(mu[1]) * sin(mu[1])
            theta_a5 = delta * sin(mu[1])**2
            #else:
            #    theta_a3 = 0.0
            #    theta_a4 = 0.0
            #    theta_a5 = 0.0
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5)
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
            theta_f0 = 1.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 0.0
            theta_g1 = 0.0
            theta_g2 = 0.0
            theta_g3 = 1.0
            return (theta_g0, theta_g1, theta_g2, theta_g3)
        elif term == "h":
            theta_h0 = 0.0
            theta_h1 = 0.0
            theta_h2 = 0.0
            theta_h3 = 1.0**2
            return (theta_h0, theta_h1, theta_h2,  theta_h3)
        elif term == "dirichlet_bc_y":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        print(term)
        dx = self.dx
        if term == "a":
            y = self.y
            q = self.q
            h = self.h
            a0 = inner(grad(y), grad(q)) * dx #- inner(grad(l), grad(v)) * dx
            a1 = y.dx(0) * q * dx #- l.dx(0) * v * dx
            a2 = y.dx(1) * q * dx #- l.dx(1) * v * dx
            a3 = h * y.dx(0) * q.dx(0) * dx #- h * l.dx(0) * v.dx(0) * dx
            a4 = h * y.dx(0) * q.dx(1) * dx + h * y.dx(1) * q.dx(0) * dx #- h * l.dx(0) * v.dx(1) * dx + h * l.dx(1) * v.dx(0) * dx
            a5 = h * y.dx(1) * q.dx(1) * dx# - h * l.dx(1) * v.dx(1) * dx 
            return (a0, a1, a2, a3, a4, a5)
        elif term == "a*":
            z = self.z
            p = self.p
            h = self.h
            as0 = inner(grad(z), grad(p)) * dx
            as1 = - p.dx(0) * z * dx 
            as2 = - p.dx(1) * z * dx
            as3 = - h * p.dx(0) * z.dx(0) *  dx(domain=mesh)
            as4 = - h * p.dx(0) * z.dx(1) * dx(domain=mesh) + h * p.dx(1) * z.dx(0) *  dx(domain=mesh)
            as5 = - h * p.dx(1) * z.dx(1) *  dx
            return (as0,as1,as2,as3,as4,as5)
        elif term == "c":
            u = self.u
            q = self.q
            c0 = u * q * dx
            return (c0,)
        elif term == "c*":
            v = self.v
            p = self.p
            cs0 = v * p * dx
            return (cs0,)
        elif term == "m":
            y = self.y
            z = self.z
            m0 = y * z * dx
            return (m0,)
        elif term == "n":
            u = self.u
            v = self.v
            n0 = u * v * dx
            return (n0,)
        elif term == "f":
            q = self.q
            f0 = Constant(0.0) * q * dx
            return (f0,)
        elif term == "g":
            z = self.z
            y_d = self.y_d
            g0 = y_d * z * dx(1, domain=mesh)
            g1 = y_d * z * dx(2, domain=mesh)
            g2 = y_d * z * dx(3, domain=mesh)
            g3 = y_d * z * dx(4, domain=mesh)
            return (g0, g1, g2, g3)
        elif term == "h":
            y_d = self.y_d
            h0 = y_d * y_d * dx(1, domain=mesh)
            h1 = y_d * y_d * dx(2, domain=mesh)
            h2 = y_d * y_d * dx(3, domain=mesh)
            h3 = y_d * y_d * dx(4, domain=mesh)
            return (h0, h1, h2, h3)
        elif term == "dirichlet_bc_y":
            bc0 = [DirichletBC(self.V.sub(0), Constant(1.0), self.boundaries, 1),
                   DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 2)]
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = [DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 2)]
            return (bc0,)
        elif term == "inner_product_y":
            y = self.y
            z = self.z
            x0 = inner(grad(y), grad(z)) * dx
            return (x0,)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0 = u * v * dx
            return (x0,)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0 = inner(grad(p), grad(q)) * dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")


# ## 4. Main program
# 
# ### 4.1. Read the mesh for this problem
# The mesh was generated by the [data/generate_mesh_1.ipynb](data/generate_mesh_1.ipynb) notebook.

# In[ ]:


mesh = Mesh("data/squareOC_N_3594.xml")
subdomains = MeshFunction("size_t", mesh, "data/squareOC_N_3594_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/squareOC_N_3594_facet_region.xml")
print("hMax: ", mesh.hmax() )

# ### 4.2. Create Finite Element space (Lagrange P1)

scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(scalar_element, scalar_element, scalar_element)
V = FunctionSpace(mesh, element, components=["y", "u", "p"])
print("Dim: ", V.dim() )

# ### 4.3. Allocate an object of the EllipticOptimalControl class


problem = EllipticOptimalControl(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1e4,5e5), (0.0, 6.3)]
problem.set_mu_range(mu_range)

offline_mu = (2e4, 1.2)
problem.init()
problem.set_mu(offline_mu)
problem.solve()
problem.export_solution(filename="FEM_OC_Square_mu_2.4_1.2_d_2.1_alpha_0.01_STSTAD")


# ### 4.4. Prepare reduction with a reduced basis method

# In[ ]:


reduced_basis_method = ReducedBasis(problem)
reduced_basis_method.set_Nmax(20)


# ### 4.5. Perform the offline phase

# In[ ]:


lifting_mu = (2e4, 1.2)
problem.set_mu(lifting_mu)
reduced_basis_method.initialize_training_set(100)
reduced_problem = reduced_basis_method.offline()


# ### 4.6. Perform an online solve

# In[ ]:


online_mu = (2e4, 1.2)
reduced_problem.set_mu(online_mu)
reduced_solution = reduced_problem.solve()
print("Reduced output for mu =", online_mu, "is", reduced_problem.compute_output())
reduced_problem.export_solution(filename="online_solution_OC_Square_mu_2.4_1.2_d_2.1_alpha_0.01_STSTAD")

# In[ ]:


plot(reduced_solution, reduced_problem=reduced_problem, component="y")


# In[ ]:


plot(reduced_solution, reduced_problem=reduced_problem, component="u")


# In[ ]:


plot(reduced_solution, reduced_problem=reduced_problem, component="p")


# ### 4.7. Perform an error analysis

# In[ ]:


reduced_basis_method.initialize_testing_set(100)
reduced_basis_method.error_analysis()


# ### 4.8. Perform a speedup analysis

# In[ ]:


reduced_basis_method.speedup_analysis()

