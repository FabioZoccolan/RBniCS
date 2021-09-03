from dolfin import *
from multiphenics import *
from rbnics import *
from problems import *
from reduction_methods import *
from numpy import ceil, zeros, isclose 

#### SETTING PROBLEM PARABOLIC + OC


class EllipticOptimalControl(EllipticOptimalControlProblem):

    # Default initialization of members
    def __init__(self, block_V, **kwargs):
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
        (self.y, self.u, self.p) = (block_yup[0:self.Nt], block_yup[self.Nt:2*self.Nt], block_yup[2*self.Nt:3*self.Nt]) 
        block_zvq = BlockTestFunction(block_V)
        (self.z, self.v, self.q) = (block_zvq[0:self.Nt], block_zvq[self.Nt:2*self.Nt], block_zvq[2*self.Nt:3*self.Nt]) 
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Regularization coefficient
        self.alpha = 0.01
        self.y_d = Constant(1.0)
       
        
        
        
        self.y_0 = Expression("1.0", degree=1, domain=mesh) #self.y_0 = Expression("1.0-1.0*(x[0]==0)-1.0*( x[0] <= 1)*(x[1]==0)-1.0*(x[0] <= 1)*( x[1]==1) ", degree=1, domain=mesh)
        
        #Boundary Conditions
        self.bc1 = Constant(1.0)
        
        self.bc2 = Expression("0.0 + 1.0*(x[0] == 0.0)*(x[1] == 0.25)", degree=1, domain= mesh)
        
        
        
        
        
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "umfpack"
        })

    # Return custom problem name
    def name(self):
        return "Parabolic_OCSquarePOD1_h_0.025_mu_2e4_1.2_alpha_0.01_d_2.1"


    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term in ("a", "a*"):
            theta_a0 = 1/(mu[0])
            theta_a1 = cos(mu[1])
            theta_a2 = sin(mu[1])
            theta_a3 = 1.0
            return (theta_a0, theta_a1, theta_a2, theta_a3)
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
            theta_f0 = 1.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = 1.
            return (theta_g0, )
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
        ds = self.ds
        dt = self.dt
        if term == "a":
            y = self.y
            q = self.q
            
            
            a0_0 = zeros((Nt, Nt), dtype=object)
            a1_0 = zeros((Nt, Nt), dtype=object)
            a2_0 = zeros((Nt, Nt), dtype=object)
            
            m_0 = zeros((Nt, Nt), dtype=object)
           
            for i in range(Nt):
                a0_0[i, i] =  dt*inner(grad(y[i]), grad(q[i]))*dx                 
                a1_0[i, i] =  dt*y[i].dx(0)*q[i]*dx 
                a2_0[i, i] =  dt*y[i].dx(1)*q[i]*dx
                      
                m_0[i, i] =  inner(y[i],q[i])*dx 
                
            for i in range(Nt-1):
                m_0[i+1, i] = - inner(y[i],q[i+1])*dx 
               
            a0 = [[0, 0, 0], [0, 0, 0], [a0_0, 0, 0]]
            a1 = [[0, 0, 0], [0, 0, 0], [a1_0, 0, 0]]
            a2 = [[0, 0, 0], [0, 0, 0], [a2_0, 0, 0]]
            
            a3 = [[0, 0, 0], [0, 0, 0], [m_0, 0, 0]]
            
            return (BlockForm(a0), BlockForm(a1), BlockForm(a2),  BlockForm(a3))
        elif term == "a*":
            z = self.z
            p = self.p
            
            
            as0_0 = zeros((Nt, Nt), dtype=object)
            as1_0 = zeros((Nt, Nt), dtype=object)
            as2_0 = zeros((Nt, Nt), dtype=object)
            
            ms_0 = zeros((Nt, Nt), dtype=object)
            
            for i in range(Nt):
                as0_0[i, i] =  dt*inner(grad(p[i]), grad(z[i]))*dx                 
                as1_0[i, i] =  -dt*p[i].dx(0)*z[i]*dx 
                as2_0[i, i] =  -dt*p[i].dx(1)*z[i]*dx
                          
                ms_0[i, i] =  inner(p[i], z[i])*dx 
               
            for i in range(Nt-1):
                ms_0[i+1, i] = - inner(p[i+1],z[i])*dx 
                
            as0 = [[0, 0, 0], [0, 0, 0], [as0_0, 0, 0]]
            as1 = [[0, 0, 0], [0, 0, 0], [as1_0, 0, 0]]
            as2 = [[0, 0, 0], [0, 0, 0], [as2_0, 0, 0]]
            
            
            as3 = [[0, 0, 0], [0, 0, 0], [ms_0, 0, 0]]
            
            return (BlockForm(as0), BlockForm(as1), BlockForm(as2),  BlockForm(as3))
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
            h = self.h
            m0_0 = zeros((Nt, Nt), dtype=object)
            
            for i in range(Nt):
                m0_0[i,i] = + dt*inner(y[i], z[i])*dx
                
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
            y_0 = self.y_0
            h = self.h
            f0_0 = zeros(Nt, dtype=object)
            
            f0_0[0] = inner(y_0, q[0])*dx 
            
            f0 = [0, 0, f0_0] 
            
            return (BlockForm(f0),)
        elif term == "g":
            y_d = self.y_d
            z = self.z
            h = self.h
            g0_0 = zeros(Nt, dtype=object)
            
            for i in range(Nt):
                g0_0[i] = dt* y_d * z[i] * dx(4)
                
            g0 = [g0_0, 0, 0]
            
            return (BlockForm(g0),)
        elif term == "h":
            y_d = self.y_d
            h0 = y_d * y_d * dx(3, domain=mesh) + y_d * y_d * dx(4, domain=mesh) 
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = BlockDirichletBC([[[DirichletBC(self.V.sub(i), self.bc1, self.boundaries, 1),
                                      DirichletBC(self.V.sub(i), self.bc2, self.boundaries, 2)] for i in range(0, Nt)], None, None])
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = BlockDirichletBC([None, None, [[DirichletBC(self.V.sub(i), Constant(0.0), self.boundaries, 1),
                                                  DirichletBC(self.V.sub(i), Constant(0.0), self.boundaries, 2)] for i in range(2*Nt, 3*Nt)]] )
         
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
                x0_u[i, i] = inner(u[i], v[i])*dx
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

##### MAIN PROGRAM

# Mesh
mesh = Mesh("data/squareOC1_h_0.025.xml")
subdomains = MeshFunction("size_t", mesh, "data/squareOC1_h_0.025_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/squareOC1_h_0.025_facet_region.xml")
print("hMax: ", mesh.hmax() )

# Create Finite Element space (Lagrange P1)
T = 3.0
dt = 0.5
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
element = BlockElement([scalar_element]*Nt + [scalar_element]*Nt + [scalar_element]*Nt)
components = ["y"]*Nt + ["u"]*Nt + ["p"]*Nt
block_V = BlockFunctionSpace(mesh, element, components=[*components]) #restrict = [*y_restrict, *u_restrict, *p_restrict]


print("Dim: ", block_V.dim() )


# 3. Allocate an object of the EllipticOptimalControl class
elliptic_optimal_control = EllipticOptimalControl(block_V, subdomains=subdomains, boundaries=boundaries, T=T, dt=dt, Nt=Nt)
mu_range =  [(1e4,5e5), (0.01, 6.3)]
elliptic_optimal_control.set_mu_range(mu_range)

offline_mu = (2e4, 1.2)
elliptic_optimal_control.init()
elliptic_optimal_control.set_mu(offline_mu)
elliptic_optimal_control.solve() 
elliptic_optimal_control.export_solution(filename="FEM_Par_OCSquare1_h_0.025_mu_2e4_1.2_alpha_0.01_d_2.1")


# ### Prepare reduction with a reduced basis method

# In[ ]:



pod_galerkin_method = PODGalerkin(elliptic_optimal_control)
pod_galerkin_method.set_Nmax(20)




# ### Perform the offline phase

# In[ ]:


lifting_mu = (2e4, 1.2)
elliptic_optimal_control.set_mu(lifting_mu)
pod_galerkin_method.initialize_training_set(100) #100
reduced_elliptic_optimal_control = pod_galerkin_method.offline()


# ### 4.6. Perform an online solve

# In[ ]:


online_mu = (2e4, 1.2)
reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_solution = reduced_elliptic_optimal_control.solve() #put the number of basis the I wanted (3)
print("Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())
reduced_elliptic_optimal_control.export_solution(filename="online_solution_Par_OCSquare1_h_0.025_mu_2e4_1.2_alpha_0.01_d_2.1")

# In[ ]:




# ### 4.7. Perform an error analysis

# In[ ]:

pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
