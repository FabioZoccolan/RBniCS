from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *


# ### Affine Decomposition



@OnlineStabilization()
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
        self.y_d = Constant(1.0)
        self.bc1 = Constant(1.0)
        
        self.delta = 2.1
        self.h = CellDiameter(V.mesh())
        
        self.bc2 = Expression("0.0 + 1.0*(x[0] == 0.0)*(x[1] == 0.25)", element=self.V.ufl_element())
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "mumps"
        })

    # Return custom problem name
    def name(self):
        return "AdvectionOCSquarePOD-2_h_0.049_STAB_mu_2.4_1.2_alpha_0.01_allsquare"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term in ("a", "a*"):
            theta_a0 = 1/(mu[0])
            theta_a1 = cos(mu[1])
            theta_a2 = sin(mu[1])
            if self.stabilized:
                delta = self.delta
                theta_a3 = delta * cos(mu[1])**2
                theta_a4 = delta * cos(mu[1]) * sin(mu[1])
                theta_a5 = delta * sin(mu[1])**2
            else:
                theta_a3 = 0.0
                theta_a4 = 0.0
                theta_a5 = 0.0
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5)
        elif term in ("c", "c*"):
            theta_c0 = 1.0
            if self.stabilized:
               delta = self.delta
               theta_c1 = delta * cos(mu[1])
               theta_c2 = delta * sin(mu[1])
            else:
               theta_c1 = 0.0
               theta_c2 = 0.0
            return (theta_c0,theta_c1, theta_c2)
        elif term == "m":
            theta_m0 = 1.0
            if self.stabilized:
               delta = self.delta
               theta_m1 = delta * cos(mu[1])
               theta_m2 = delta * sin(mu[1])           
            else:
               theta_m1 = 0.0
               theta_m2 = 0.0
            return (theta_m0,theta_m1, theta_m2)
        elif term == "n":
            theta_n0 = self.alpha
            return (theta_n0,)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.0
            if self.stabilized:
               delta = self.delta
               theta_g1 = delta * cos(mu[1])
               theta_g2 = delta * sin(mu[1])
            else:
               theta_g1 = 0.0
               theta_g2 = 0.0
            return (theta_g0, theta_g1, theta_g2)
        elif term == "h":
            theta_h0 = 0.0
            theta_h1 = 1.0**2
            return (theta_h0, theta_h1)
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
            h = self.h
            y = self.y
            q = self.q
            a0 = inner(grad(y), grad(q)) * dx
            a1 = y.dx(0) * q * dx 
            a2 = y.dx(1) * q * dx
            a3 = h * y.dx(0) * q.dx(0) * dx
            a4 = h * y.dx(0) * q.dx(1) * dx + h * y.dx(1) * q.dx(0) * dx 
            a5 = h * y.dx(1) * q.dx(1) * dx
            return (a0, a1, a2, a3, a4, a5)
        elif term == "a*":
            z = self.z
            p = self.p
            h = self.h
            as0 = inner(grad(z), grad(p)) * dx
            as1 = - p.dx(0) * z * dx
            as2 = - p.dx(1) * z * dx
            as3 = h * p.dx(0) * z.dx(0) * dx 
            as4 = h * p.dx(0) * z.dx(1) * dx + h * p.dx(1) * z.dx(0) * dx
            as5 = h * p.dx(1) * z.dx(1) * dx 
            return (as0, as1, as2, as3, as4, as5)
        elif term == "c":
            u = self.u
            q = self.q
            h = self.h
            c0 = u * q * dx
            c1 = h * u * q.dx(0) * dx
            c2 = h * u * q.dx(1) * dx
            return (c0,c1, c2)
        elif term == "c*":
            v = self.v
            p = self.p
            h = self.h
            cs0 = v * p * dx
            cs1 = Constant(0.0) * h * v * p.dx(0) * dx
            cs2 = Constant(0.0)  * h * v * p.dx(1) * dx
            return (cs0,cs1, cs2)
        elif term == "m":
            y = self.y
            z = self.z
            h = self.h
            m0 = y * z * dx
            m1 = - h * y * z.dx(0) * dx
            m2 = - h * y * z.dx(1) * dx
            return (m0,m1, m2)
        elif term == "n":
            u = self.u
            v = self.v
            h = self.h
            n0 = u * v * dx
            return (n0,)
        elif term == "f":
            q = self.q
            f0 = Constant(0.0) * q * dx
            return (f0,)
        elif term == "g":
            z = self.z
            y_d = self.y_d
            h = self.h
            g0 = y_d * z * dx
            g1 = - h * y_d * z.dx(0) * dx
            g2 = - h * y_d * z.dx(1) * dx
            return (g0, g1, g2)
        elif term == "h":
            y_d = self.y_d
            h0 = y_d * y_d * dx(1, domain=mesh)
            h1 = y_d * y_d * dx(2, domain=mesh)
            return (h0, h1)
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


mesh = Mesh("data/square_h_0.049.xml")
subdomains = MeshFunction("size_t", mesh, "data/square_h_0.049_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/square_h_0.049_facet_region.xml")
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
problem.export_solution(filename="FEM_OC_Square2_h_0.049_STAB_mu_2.4_1.2_alpha_0.01_allsquare")


## ### 4.4. Prepare reduction with a reduced basis method

# In[ ]:

pod_galerkin_method = PODGalerkin(problem)
pod_galerkin_method.set_Nmax(20)

# ### 4.5. Perform the offline phase

# In[ ]:


lifting_mu = (2e4, 1.2)
problem.set_mu(lifting_mu)
pod_galerkin_method.initialize_training_set(100)
reduced_elliptic_optimal_control = pod_galerkin_method.offline()

# ### 4.6. Perform an online solve

# In[ ]:


online_mu = (2e4, 1.2)
reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_solution = reduced_elliptic_optimal_control.solve()
print("Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())
reduced_elliptic_optimal_control.export_solution(filename="FEM_OC_Square2_h_0.049_STAB_mu_2.4_1.2_alpha_0.01_allsquare")

# ### 4.7. Perform an error analysis

# In[ ]:


pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()


# ### 4.8. Perform a speedup analysis

# In[ ]:

pod_galerkin_method.speedup_analysis()

