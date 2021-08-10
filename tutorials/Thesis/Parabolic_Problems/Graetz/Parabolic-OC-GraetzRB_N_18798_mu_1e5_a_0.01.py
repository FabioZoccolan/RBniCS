from dolfin import *
from multiphenics import *
from rbnics import *
from problems import *
from reduction_methods import *
from numpy import ceil , zeros #isclose, 

"""### 3. Affine Decomposition

For this problem the affine decomposition is straightforward.
"""

class EllipticOptimalControl(EllipticOptimalControlProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticOptimalControlProblem.__init__(self, block_V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.T =  kwargs["T"]
        self.dt =  kwargs["dt"]
        self.Nt =  kwargs["Nt"]
        block_yup = BlockTrialFunction(block_V)
        #block_yup[0] = (Expression("Constant(1.0)", element = self.V.sub(0).ufl.element()),)+ block_yup[1:] #,domain=mesh)
        (self.y, self.u, self.p) = (block_yup[0:self.Nt], block_yup[self.Nt:2*self.Nt], block_yup[2*self.Nt:3*self.Nt]) 
        block_zvq = BlockTestFunction(block_V)
        (self.z, self.v, self.q) = (block_zvq[0:self.Nt], block_zvq[self.Nt:2*self.Nt], block_zvq[2*self.Nt:3*self.Nt]) 
        self.dx = Measure("dx")(subdomain_data=subdomains)
        self.ds = Measure("ds")(subdomain_data=boundaries)
        # Regularization coefficient
        self.alpha = 0.01
        self.y_d = Constant(1.0)
        # Store the velocity expression
        self.vel = Expression("x[1] * (1 - x[1])", degree=2, domain=mesh)
        #self.lifting = Expression('((x[0] >= 1 && x[0] <= 2) && (x[1] == 1.0 || x[1]== 0.0) ) ? 1. : 0.', degree=1, domain=mesh)
        
        ####self.y_0 = Expression(("1*(x[0] == 0)"), degree=2, domain=mesh)
        
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "umfpack"
        })

    # Return custom problem name
    def name(self):
        return "Parabolic_OCGraetzRB_N_18798_mu_1e5_alpha_0.01"
        
        # Return stability factor
    def get_stability_factor_lower_bound(self):
        return 1.

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
            return (theta_n0,)
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
        y = self.y
        q = self.q
        #y = (Expression("1.0", element=self.V.sub(0).ufl_element(), domain=mesh),) + y[1:] CONTROLLARE
        if term == "a":
            vel = self.vel
            a0_0 = zeros((Nt, Nt), dtype=object)
            a1_0 = zeros((Nt, Nt), dtype=object)
            #y = (1,) + y[1:] #AGGIUNTO IO, INITIAL CONDITIONS
            for i in range(Nt):
                a0_0[i, i] = dt*inner(grad(y[i]), grad(q[i]))*dx #a0 = inner(grad(y), grad(q)) * dx   
                a0_0[i, i] = dt*inner(grad(y[i]), grad(q[i]))*dx
                a1_0[i, i] =  inner(y[i],q[i])*dx + dt*vel*y[i].dx(0)*q[i]*dx #a1 = vel * y.dx(0) * q * dx
                a1_0[i, i] =  inner(y[i],q[i])*dx + dt*vel*y[i].dx(0)*q[i]*dx 
            for i in range(Nt-1):
                a1_0[i+1,i] = - inner(y[i], q[i+1])*dx
            a0 = [[0, 0, 0], [0, 0, 0], [a0_0, 0, 0]]
            a1 = [[0, 0, 0], [0, 0, 0], [a1_0, 0, 0]]
            return (BlockForm(a0), BlockForm(a1))
        elif term == "a*":
            z = self.z
            p = self.p
            vel = self.vel
            as0_0 = zeros((Nt, Nt), dtype=object)
            as1_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                as0_0[i, i] = dt*inner(grad(z[i]), grad(p[i]))*dx(1) #as0 = inner(grad(z), grad(p)) * dx
                as1_0[i, i] = inner(z[i],p[i])*dx - dt*vel*p[i].dx(0)*z[i]*dx #as1 = -vel * p.dx(0) * z * dx
            for i in range(Nt-1):
                as1_0[i+1, i] = - inner(p[i+1], z[i])*dx
            as0 = [[0, 0, as0_0], [0, 0, 0], [0, 0, 0]]
            as1 = [[0, 0, as1_0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(as0), BlockForm(as1))
        elif term == "c":
            u = self.u
            q = self.q
            c0_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                c0_0[i,i] = + dt*inner(u[i], q[i])*dx
            c0 = [[0, 0, 0], [0, 0, 0], [0, c0_0, 0]]
            return(BlockForm(c0),)
        elif term == "c*":
            v = self.v
            p = self.p
            cs0_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                cs0_0[i,i] = + dt*inner(p[i], v[i])*dx
            cs0 = [[0, 0, 0], [0, 0, cs0_0], [0, 0, 0]]
            return(BlockForm(cs0),)
        elif term == "m":
            y = self.y
            z = self.z
            m0_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                m0_0[i,i] = dt*inner(y[i], z[i])*dx
            m0 = [[m0_0, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(m0),)
        elif term == "n":
            u = self.u
            v = self.v
            n0_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                n0_0[i,i] = dt*inner(u[i], v[i])*dx #*ds(2)
            n0 = [[0, 0, 0], [0, n0_0, 0], [0, 0, 0]]
            return (BlockForm(n0),)
        elif term == "f":
            q = self.q
            f0_0 = zeros(Nt, dtype=object)
            f0_0[0] = inner(Constant(0.0), q[0])*dx #f0_0[0] = inner(y_0, q[0])*dxfor i in range(2*Nt, 3*Nt)
            f0 = [0, 0, f0_0]
            return (BlockForm(f0),)
        elif term == "g":
            y_d = self.y_d
            z = self.z
            g0_0 = zeros(Nt, dtype=object)
            for i in range(Nt):
                g0_0[i] = dt * y_d * z[i] * dx(3) + dt*y_d * z[i] * dx(4)
            g0 = [g0_0, 0, 0]
            return (BlockForm(g0),)
        elif term == "h":
            y_d = self.y_d
            h0 = y_d * y_d * dx(3, domain=mesh) + y_d * y_d * dx(4, domain=mesh)  #RICONTROLLARE
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = BlockDirichletBC([[[DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 1),
                   DirichletBC(block_V.sub(i), Constant(1.0), self.boundaries, 2),
                   DirichletBC(block_V.sub(i), Constant(1.0), self.boundaries, 4),
                   DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 5),
                   DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 6)] for i in range(0, Nt)], None, None])
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = BlockDirichletBC([None, None, [[DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 1),
                   DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 2),
                   #DirichletBC(block.V.sub(2), Constant(0.0), self.boundaries, 3), #RICONTROLLARE
                   DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 4),
                   DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 5),
                   DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 6)] for i in range(2*Nt, 3*Nt)]] )
         
            return (bc0,)
        elif term == "inner_product_y":
            y = self.y
            z = self.z
            x0_y = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                x0_y[i, i] = inner(grad(y[i]), grad(z[i]))*dx
            x0 = [[x0_y, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(x0),)
        elif term == "inner_product_u":
            u = self.u
            v = self.v
            x0_u = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                x0_u[i, i] = inner(u[i], v[i])*ds(2)
            x0 = [[0, 0, 0], [0, x0_u, 0], [0, 0, 0]]
            return (BlockForm(x0),)
        elif term == "inner_product_p":
            p = self.p
            q = self.q
            x0_p = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                x0_p[i, i] = inner(grad(p[i]), grad(q[i]))*dx
            x0 = [[0, 0, 0], [0, 0, 0], [0, 0, x0_p]]
            return (BlockForm(x0),)
        else:
            raise ValueError("Invalid term for assemble_operator().")

"""## 4. Main program

### 4.1. Read the mesh for this problem
The mesh was generated by the [data/generate_mesh_2.ipynb](https://colab.research.google.com/github/RBniCS/RBniCS/blob/open-in-colab/tutorials/13_elliptic_optimal_control/data/generate_mesh_2.ipynb
) notebook.
"""

# 1. Mesh
mesh = Mesh("data/graetzOC_N_18798.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetzOC_N_18798_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetzOC_N_18798_facet_region.xml")
print("hMax: ", mesh.hmax() )

# 2 Create Finite Element space (Lagrange P1)
T = 4.5
dt = 0.5
Nt = int(ceil(T/dt))

# BOUNDARY RESTRICTIONS #
y_restrict = []
u_restrict = []
p_restrict = []
for i in range(Nt):
    y_restrict.append(None)
    u_restrict.append(None)
    p_restrict.append(None)

# FUNCTION SPACES #
scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = BlockElement([scalar_element]*Nt + [scalar_element]*Nt + [scalar_element]*Nt)
components = ["y"]*Nt + ["u"]*Nt + ["p"]*Nt
block_V = BlockFunctionSpace(mesh, element, restrict = [*y_restrict, *u_restrict, *p_restrict], components=[*components])


print("Dim: ", block_V.dim() )


# 3. Allocate an object of the EllipticOptimalControl class
elliptic_optimal_control = EllipticOptimalControl(block_V, subdomains=subdomains, boundaries=boundaries, T=T, dt=dt, Nt=Nt)
mu_range =  [(0.01, 1e6), (0.5, 4.0)]
elliptic_optimal_control.set_mu_range(mu_range)

offline_mu = (6.0, 1.0)
elliptic_optimal_control.init()
elliptic_optimal_control.set_mu(offline_mu)
elliptic_optimal_control.solve()
elliptic_optimal_control.export_solution(filename="FEM_Par_OCGraetz_N_18798_mu_1e5_alpha_0.01")


# ### 4.4. Prepare reduction with a reduced basis method

# In[ ]:


reduced_basis_method = ReducedBasis(elliptic_optimal_control)
reduced_basis_method.set_Nmax(20)


# ### 4.5. Perform the offline phase

# In[ ]:


lifting_mu = (6.0, 1.0)
elliptic_optimal_control.set_mu(lifting_mu)
reduced_basis_method.initialize_training_set(100)
reduced_elliptic_optimal_control = reduced_basis_method.offline()


# ### 4.6. Perform an online solve

# In[ ]:


online_mu = (6.0, 1.0)
reduced_problem.set_mu(online_mu)
reduced_solution = reduced_problem.solve()
print("Reduced output for mu =", online_mu, "is", reduced_problem.compute_output())
reduced_problem.export_solution(filename="online_solution_Par_OCGraetz_N_18798_mu_1e5_alpha_0.01")

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

