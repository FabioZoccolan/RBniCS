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
class AdvectionDominated(EllipticCoerciveProblem):

    # Default initialization of members
    def __init__(self, block_V, **kwargs): #ho messo block_V invece che V
        # Call the standard initialization
        EllipticCoerciveProblem.__init__(self, block_V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.T =  kwargs["T"]
        self.dt =  kwargs["dt"]
        self.Nt =  kwargs["Nt"]
        block_u = BlockTrialFunction(block_V)
        (self.u) = (block_u[0:self.Nt])
        block_v = BlockTestFunction(block_V)
        (self.v) = (block_v[0:self.Nt])
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        # Regularization coefficient
        # Store the velocity expression
        self.vel = Expression("x[1] * (1 - x[1])", degree=1, domain=mesh)
        
        self.delta = 1.0
        
        self.h = CellDiameter(block_V.mesh())
        self.y_0 = Expression("1.0-1.0*(x[0]==0)-1.0*( x[0] <= 1)*(x[1]==0)-1.0*(x[0] <= 1)*( x[1]==1) ", degree=1, domain=mesh) #- 1.0*(x[0]==0)-1.0*( x[0] <= 1)*(x[1]==0)-1.0*(x[0] <= 1)*( x[1]==1)
   
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "umfpack"
        })

    # Return custom problem name
    def name(self):
        return "scratchParabolic_GraetzPOD_STAB_N_18798_mu_1e5_alpha_0.01"
        
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
            theta_a4 = 1.0
            if self.stabilized:
               theta_a5 = self.delta
            else:
               theta_a5 = 0.0
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4) #, theta_a5)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0, )
        elif term == "dirichlet_bc":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        print(term)
        dx = self.dx
        v = self.v
        if term == "a":
            u = self.u
            h = self.h
            vel = self.vel
            a0_0 = zeros((Nt, Nt), dtype=object)
            a1_0 = zeros((Nt, Nt), dtype=object)
            a2_0 = zeros((Nt, Nt), dtype=object)
            a3_0 = zeros((Nt, Nt), dtype=object)
            m_0 = zeros((Nt, Nt), dtype=object)
            m_1 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                a0_0[i, i] = dt*inner(grad(u[i]), grad(v[i]))*dx #a0 = inner(grad(y), grad(q)) * dx   
                
                a1_0[i, i] = + dt*vel*u[i].dx(0)*v[i]*dx #a1 = vel * y.dx(0) * q * dx
                a2_0[i, i] = dt * h * vel * u[i].dx(0) * v[i].dx(0) * dx(1) 
                a3_0[i, i] =  dt * h * vel * u[i].dx(0) * v[i].dx(0) * dx(2) + dt * h * vel * u[i].dx(0) * v[i].dx(0) * dx(3) + dt * h * vel * u[i].dx(0) * v[i].dx(0) * dx(4) 
                m_0[i,i] = inner(u[i],v[i])*dx
               # m_1[i,i] = h * u[i] * v[i].dx(0)*dx
            for i in range(Nt-1):
                m_0[i+1,i] = - inner(u[i], v[i+1])*dx
               # m_1[i+1,i] = -h * u[i] * v[i+1].dx(0) *dx
            a0 = [[0, 0, 0], [0, 0, 0], [a0_0, 0, 0]]
            a1 = [[0, 0, 0], [0, 0, 0], [a1_0, 0, 0]]
            a2 = [[0, 0, 0], [0, 0, 0], [a2_0, 0, 0]]
            a3 = [[0, 0, 0], [0, 0, 0], [a3_0, 0, 0]]
            a4 = [[0, 0, 0], [0, 0, 0], [m_0, 0, 0]]
            #a5 = [[0, 0, 0], [0, 0, 0], [m_1, 0, 0]]
            return (BlockForm(a0), BlockForm(a1), BlockForm(a2), BlockForm(a3), BlockForm(a4)) #,  BlockForm(a5))
        elif term == "f":
            y_0 = self.y_0
            #y_0 = Constant(1.0)
            f0_0 = zeros(Nt, dtype=object)
            f0_0[0] = y_0 * v[0] * dx 
            f0 = [0, 0, f0_0] 
            return (BlockForm(f0),)
        elif term == "dirichlet_bc":
            bc0 = BlockDirichletBC([[DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 1),
                                           DirichletBC(block_V.sub(i), Constant(1.0), self.boundaries, 2),
                                           DirichletBC(block_V.sub(i), Constant(1.0), self.boundaries, 4),
                                           DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 5),
                                           DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 6)] for i in range(0, Nt)])
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0_y = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                x0_y[i, i] = inner(grad(u[i]), grad(v[i]))*dx
            x0 = [[x0_y, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(x0),)
        else:
            raise ValueError("Invalid term for assemble_operator().")

"""## 4. Main program

### 4.1. Read the mesh for this problem
The mesh was generated by the [data/generate_mesh_2.ipynb](https://colab.research.google.com/github/RBniCS/RBniCS/blob/open-in-colab/tutorials/13_elliptic_optimal_control/data/generate_mesh_2.ipynb
) notebook.
"""

# 1. Mesh
mesh = Mesh("data/graetzOC_h_0.029.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetzOC_h_0.029_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetzOC_h_0.029_facet_region.xml")
print("hMax: ", mesh.hmax() )

# 2 Create Finite Element space (Lagrange P1)
T = 7.2
dt = 0.4
Nt = int(ceil(T/dt))

# BOUNDARY RESTRICTIONS #
#y_restrict = []
#u_restrict = []
#p_restrict = []
#for i in range(Nt):
#    y_restrict.append(None)
  #  u_restrict.append(None)
   # p_restrict.append(None)

# FUNCTION SPACES #
scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = BlockElement([scalar_element]*Nt)
components = ["u"]*Nt 
block_V = BlockFunctionSpace(mesh, element, components=[*components]) #restrict = [*y_restrict, *u_restrict, *p_restrict]


print("Dim: ", block_V.dim() )


# 3. Allocate an object of the EllipticOptimalControl class
problem = AdvectionDominated(block_V, subdomains=subdomains, boundaries=boundaries, T=T, dt=dt, Nt=Nt)
mu_range =  [(0.01, 1e6), (0.5, 4.0)]
problem.set_mu_range(mu_range)

offline_mu = (1e5, 1.0)
problem.init()
problem.set_mu(offline_mu)
problem.solve()
problem.export_solution(filename="FEM_Par_OCGraetz_STAB_N_18798_mu_1e5_alpha_0.01")


# ### 4.4. Prepare reduction with a reduced basis method

# In[ ]:



reduction_method = PODGalerkin(problem)
reduction_method.set_Nmax(3)
reduction_method.set_tolerance(1e-7) #AGGIUNTO



# ### 4.5. Perform the offline phase

# In[ ]:


lifting_mu = (1e5, 1.0)
problem.set_mu(lifting_mu)
reduction_method.initialize_training_set(200)
reduced_problem = reduction_method.offline()

# ### 4.6. Perform an online solve

# In[ ]:


online_mu = (1e5, 1.0)
reduced_problem.set_mu(online_mu)
reduced_problem.solve(online_stabilization=True)
reduced_problem.export_solution(filename="online_solution_Par_Graetz_STAB_N_5503_with_stabilization_mu_10.0")
reduced_problem.solve(online_stabilization=False)
reduced_problem.export_solution(filename="online_solution_Par_Graetz_STAB_N_5503_without_stabilization_mu_10.0")

# ### 4.7. Perform an error analysis

# 7. Perform an error analysis
reduction_method.initialize_testing_set(100)
reduction_method.error_analysis(online_stabilization=True, filename="error_analysis_Par_Graetz_STAB_N_5503_with_stabilization_mu_10.0")
reduction_method.error_analysis(online_stabilization=False, filename="error_analysis_Graetz_STAB_N_5503_without_stabilization_mu_10.0")

# 8. Perform a speedup analysis
reduction_method.speedup_analysis(online_stabilization=True, filename="speedup_analysis_Par_Graetz_STAB_N_5503_with_stabilization_mu_10.0")
reduction_method.speedup_analysis(online_stabilization=False, filename="speedup_analysis_Par_Graetz_STAB_N_5_without_stabilization_mu_10.0") 
