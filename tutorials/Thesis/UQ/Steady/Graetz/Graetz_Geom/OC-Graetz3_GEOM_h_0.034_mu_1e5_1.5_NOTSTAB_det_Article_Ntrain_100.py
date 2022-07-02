from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *
from rbnics.sampling.distributions import *
from reduction_methods import *
from sampling.distributions import *
from sampling.weights import *


@WeightedUncertaintyQuantification()
@OnlineStabilization()
@PullBackFormsToReferenceDomain()
@ShapeParametrization(
    ("x[0]", "x[1]"), # subdomain 1
    ("mu[1]*(x[0] - 1) + 1", "x[1]"), # subdomain 2
    ("mu[1]*(x[0] - 1) + 1", "x[1]"), # subdomain 3
    ("mu[1]*(x[0] - 1) + 1", "x[1]"), # subdomain 4
)
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
        self.y_d = Constant(1.0)
        
        self.delta = 1.0
        
        self.h = CellDiameter(V.mesh())
        
        # Store the velocity expression
        self.vel = Expression("x[1] * (1 - x[1])", element=self.V.sub(0).ufl_element())
        #self.lifting = Expression('((x[0] >= 1 && x[0] <= 2) && (x[1] == 1.0 || x[1]== 0.0) ) ? 1. : 0.', degree=1, domain=mesh)
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "mumps"
        })

    # Return custom problem name
    def name(self):
        return "Article2/Beta0503_0503/OCGraetzPOD3_GEOM_NotSTAB_h_0.034_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100"

   # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term in ("a", "a*"):
            theta_a0 = 1/(mu[0])
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
            
            return (theta_n0, ) 
        elif term == "f":
            theta_f0 = 0.0
            return (theta_f0, )
        elif term == "g":
            theta_g0 = 1.
            
            return (theta_g0,) 
        elif term == "h":
            theta_h0 = 1.
            return (theta_h0,)
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
            vel = self.vel
            h = self.h
            a0 = inner(grad(y), grad(q)) * dx
            a1 = vel * y.dx(0) * q * dx
            return (a0, a1)
        elif term == "a*":
            z = self.z
            p = self.p
            vel = self.vel    
            h = self.h
            as0 = inner(grad(z), grad(p)) * dx
            as1 = -vel * p.dx(0) * z * dx
            return (as0, as1)
        elif term == "c":
            u = self.u
            q = self.q
            h = self.h
            c0 = u * q * dx
           
            return (c0,)
        elif term == "c*":
            v = self.v
            p = self.p
            h = self.h
            cs0 = v * p * dx
           
            return (cs0,)
        elif term == "m":
            y = self.y
            z = self.z
            h = self.h
            
            m0 = y * z * dx(3) + y * z * dx(4)
           
            return (m0,)
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
            y_d = self.y_d
            z = self.z
            h = self.h
            g0 = y_d * z * dx(3) + y_d * z * dx(4)
            
            return (g0,)
        elif term == "h":
            y_d = self.y_d
            h0 = y_d * y_d * dx(3, domain=mesh) + y_d * y_d * dx(4, domain=mesh)  #RICONTROLLARE
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = [DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V.sub(0), Constant(1.0), self.boundaries, 2),
                   DirichletBC(self.V.sub(0), Constant(1.0), self.boundaries, 4),
                   DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 5),
                   DirichletBC(self.V.sub(0), Constant(0.0), self.boundaries, 6)]
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = [DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 1),
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 2),
                   #DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 3), #RICONTROLLARE
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 4),
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 5),
                   DirichletBC(self.V.sub(2), Constant(0.0), self.boundaries, 6)]
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


"""## 4. Main program

### 4.1. Read the mesh for this problem
The mesh was generated by the [data/generate_mesh_2.ipynb](https://colab.research.google.com/github/RBniCS/RBniCS/blob/open-in-colab/tutorials/13_elliptic_optimal_control/data/generate_mesh_2.ipynb
) notebook.
"""

mesh = Mesh("data/graetzOC_h_0.034.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetzOC_h_0.034_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetzOC_h_0.034_facet_region.xml")
print("hMax: ", mesh.hmax() )


"""### 4.2. Create Finite Element space (Lagrange P1)"""

scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = MixedElement(scalar_element, scalar_element, scalar_element)
V = FunctionSpace(mesh, element, components=["y", "u", "p"])
print("Dim: ", V.dim() )

"""### 4.3. Allocate an object of the EllipticOptimalControl class"""

problem = EllipticOptimalControl(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1,1e5), (0.5, 1.5)]
problem.set_mu_range(mu_range)

# Prepare reduction with a POD-Galerkin method
pod_galerkin_method = PODGalerkin(problem)
pod_galerkin_method.set_Nmax(20)

# Perform the offline phase
pod_galerkin_method.initialize_training_set(100) #, sampling=BetaDistribution(beta_a, beta_b), weight=BetaWeight(beta_a, beta_b))
reduced_elliptic_optimal_control = pod_galerkin_method.offline()

# Perform an offline solve
offline_mu = (1e5,1.5) #(10**4.8, 3.3)
problem.set_mu(offline_mu)
problem.solve()
problem.export_solution(filename="FEM_OCGraetz3_GEOM_NotSTAB_h_0.034_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")
print("OFFLINE SOLUTION: Full order output for mu =", offline_mu, "is", problem.compute_output())


# ### 4.6. Perform an online solve

# In[ ]:


online_mu = (1e5,1.5) #(10**4.8, 3.3)

'''
reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_solution = reduced_elliptic_optimal_control.solve(online_stabilization=False) 
print("ONLINE SOLUTION - NOT ONLINE STAB: Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())
reduced_elliptic_optimal_control.export_solution(filename="online_solution_OCGraetz3_GEOM_h_0.034_OffSTAB_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")
reduced_elliptic_optimal_control.export_error(filename="online_error_OCGraetz3_GEOM_h_0.034_OffSTAB_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")

reduced_solution = reduced_elliptic_optimal_control.solve(online_stabilization=True) 
print("ONLINE SOLUTION - ONLINE STAB: Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())
reduced_elliptic_optimal_control.export_solution(filename="online_solution_OCGraetz3_GEOM_h_0.034_OffONSTAB_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")
reduced_elliptic_optimal_control.export_error(filename="online_error_OCGraetz3_GEOM_h_0.034_OffONSTAB_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")



# ### 4.7. Perform an error analysis

# In[ ]:


pod_galerkin_method.initialize_testing_set(100) #, sampling=BetaDistribution(beta_a, beta_b), enable_import = True)

print("\n----------------------------------------OFFLINE STABILIZATION ERROR ANALYSIS BEGINS-------------------------------------------------\n")

pod_galerkin_method.error_analysis(online_stabilization=False, filename="error_analysis_OCGraetz3_GEOM_h_0.034_OffSTAB_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")

print("\n--------------------------------------ONLINE-OFFLINE STABILIZATION ERROR ANALYSIS BEGINS--------------------------------------------\n")

pod_galerkin_method.error_analysis(online_stabilization=True, filename="error_analysis_OCGraetz3_GEOM_h_0.034_OffONSTAB_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")
'''


# ### 5. Perform a speedup analysis

# In[ ]:

print("\n-----------------------------------------OFFLINE STABILIZATION SPEED-UP ANALYSIS BEGINS----------------------------------------------\n")
print("")
pod_galerkin_method.speedup_analysis(online_stabilization=False, filename="speedup_analysis_OCGraetz3_GEOM_h_0.034_OffSTAB_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")
print("\n---------------------------------------ONLINE-OFFLINE STABILIZATION SPEED-UP ANALYSIS BEGINS------------------------------------------\n")
pod_galerkin_method.speedup_analysis(online_stabilization=True, filename="speedup_analysis_OCGraetz3_GEOM_h_0.034_OffONSTAB_mu_1e5_1.5_alpha_0.01_beta0301_0503_det_oldtest_Ntrain_100")