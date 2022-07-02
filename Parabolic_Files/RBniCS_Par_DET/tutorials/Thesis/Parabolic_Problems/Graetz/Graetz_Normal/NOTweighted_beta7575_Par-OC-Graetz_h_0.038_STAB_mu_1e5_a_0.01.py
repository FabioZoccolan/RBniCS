from dolfin import *
from multiphenics import *
from rbnics import *
from problems import *
from reduction_methods import *
from numpy import ceil, zeros, isclose 

from rbnics.sampling.distributions import *
from reduction_methods import *
from sampling.distributions import *
from sampling.weights import *


@WeightedUncertaintyQuantification()
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
        # Store the velocity expression
        self.vel = Expression("x[1] * (1 - x[1])", degree=1, domain=mesh)
        
        
        #self.y_0 = Expression("( ( x[1] == 1 || x[1]==0)*(x[0] > 1) || ( x[0]== 2 )*( x[1]>=0.8 || x[1]<=0.2 ) )? 1.0 : 0.0", degree=1, domain=mesh)
        #self.y_0 = Expression("( (x[0]== 2 ) && ( x[1]>=0.8 || x[1]<=0.2) )? 1.0 : 0.0", degree=1, domain=mesh)
        self.y_0 = Expression("0.0", degree=1, domain=mesh)
        
        self.delta = 1.0
        
        self.h = CellDiameter(block_V.mesh())
        
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "mumps"
        })
        PETScOptions.set("mat_mumps_icntl_7", 5)
        PETScOptions.set("mat_mumps_icntl_14", 98)

    # Return custom problem name
    def name(self):
        return "NOTWeighted_Beta_0209_Parabolic_OCGraetzPOD_h_0.038_STAB_mu_1e5_alpha_0.01_d_1"


    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term in ("a", "a*"):
            theta_a0 = 1/(mu[0])
            theta_a1 = 4.0
            if self.stabilized:
                delta = self.delta
                theta_a2 = delta * 4.0
            else:
                theta_a2 = 0.0
            theta_a3 = 1.0
            if self.stabilized:
               theta_a4 = self.delta
            else:
               theta_a4 = 0.0
            return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4)
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
            return (theta_n0, )
        elif term == "f":
            theta_f0 = 1.0
            if self.stabilized:
               delta = self.delta
               theta_f1 = delta
            else:
               theta_f1 = 0.0
            return (theta_f0,theta_f1)
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

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        print(term)
        dx = self.dx
        ds = self.ds
        dt = self.dt
        if term == "a":
            y = self.y
            q = self.q
            h = self.h
            vel = self.vel
            a0_0 = zeros((Nt, Nt), dtype=object)
            a1_0 = zeros((Nt, Nt), dtype=object)
            a2_0 = zeros((Nt, Nt), dtype=object)
            a3_0 = zeros((Nt, Nt), dtype=object)
            m_0 = zeros((Nt, Nt), dtype=object)
            m_1 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                a0_0[i, i] =  dt*inner(grad(y[i]), grad(q[i]))*dx                 
                a1_0[i, i] =  dt*vel*y[i].dx(0)*q[i]*dx 
                a2_0[i, i] =  dt * h * vel * y[i].dx(0) * q[i].dx(0) * dx               
                m_0[i, i] =  inner(y[i],q[i])*dx 
                m_1[i, i] =  h * inner(y[i], q[i].dx(0))*dx
            for i in range(Nt-1):
                m_0[i+1, i] = - inner(y[i],q[i+1])*dx 
                m_1[i+1, i] = - h * inner(y[i], q[i+1].dx(0))*dx
            a0 = [[0, 0, 0], [0, 0, 0], [a0_0, 0, 0]]
            a1 = [[0, 0, 0], [0, 0, 0], [a1_0, 0, 0]]
            a2 = [[0, 0, 0], [0, 0, 0], [a2_0, 0, 0]]
            
            a3 = [[0, 0, 0], [0, 0, 0], [m_0, 0, 0]]
            a4 = [[0, 0, 0], [0, 0, 0], [m_1, 0, 0]]
            return (BlockForm(a0), BlockForm(a1), BlockForm(a2),  BlockForm(a3), BlockForm(a4))
        elif term == "a*":
            z = self.z
            p = self.p
            h = self.h
            vel = self.vel
            as0_0 = zeros((Nt, Nt), dtype=object)
            as1_0 = zeros((Nt, Nt), dtype=object)
            as2_0 = zeros((Nt, Nt), dtype=object)
            as3_0 = zeros((Nt, Nt), dtype=object)
            ms_0 = zeros((Nt, Nt), dtype=object)
            ms_1 = zeros((Nt, Nt), dtype=object)
            
            
            for i in range(Nt):
                as0_0[i, i] = dt*inner(grad(z[i]), grad(p[i]))*dx 
                as1_0[i, i] = - dt*vel*p[i].dx(0)*z[i]*dx                
                as2_0[i, i] = dt*h * vel * p[i].dx(0) * z[i].dx(0) * dx
                ms_0[i, i] = + inner(z[i],p[i])*dx 
                ms_1[i, i] = - h * inner(p[i], z[i].dx(0))*dx 
            for i in range(Nt-1):
                ms_0[i, i+1] = - inner(p[i+1], z[i])*dx
                ms_1[i, i+1] = + h*inner(p[i+1], z[i].dx(0))*dx 
            as0 = [[0, 0, as0_0], [0, 0, 0], [0, 0, 0]]
            as1 = [[0, 0, as1_0], [0, 0, 0], [0, 0, 0]]
            as2 = [[0, 0, as2_0], [0, 0, 0], [0, 0, 0]]
            as3 = [[0, 0, ms_0], [0, 0, 0], [0, 0, 0]]
            as4 = [[0, 0, ms_1], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(as0), BlockForm(as1), BlockForm(as2),  BlockForm(as3), BlockForm(as4))
        elif term == "c":
            u = self.u
            q = self.q
            h = self.h
            c0_0 = zeros((Nt, Nt), dtype=object)
            c1_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                c0_0[i,i] = + dt*inner(u[i], q[i])*dx
                c1_0[i,i] = + dt*h * inner(u[i], q[i].dx(0)) * dx
            c0 = [[0, 0, 0], [0, 0, 0], [0, c0_0, 0]]
            c1 = [[0, 0, 0], [0, 0, 0], [0, c1_0, 0]]
            return(BlockForm(c0),BlockForm(c1))
        elif term == "c*":
            v = self.v
            p = self.p
            h = self.h
            cs0_0 = zeros((Nt, Nt), dtype=object)
            cs1_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                cs0_0[i,i] = + dt*inner(p[i], v[i])*dx
                cs1_0[i,i] = Constant(0.0) * dt* h * inner(p[i], v[i])*dx
            cs0 = [[0, 0, 0], [0, 0, cs0_0], [0, 0, 0]]
            cs1 = [[0, 0, 0], [0, 0, cs1_0], [0, 0, 0]]
            return(BlockForm(cs0),BlockForm(cs1))
        elif term == "m":
            y = self.y
            z = self.z
            h = self.h
            m0_0 = zeros((Nt, Nt), dtype=object)
            m1_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                m0_0[i,i] = dt*inner(y[i], z[i])*dx(3) + dt*inner(y[i], z[i])*dx(4)
                m1_0[i,i] = - dt * h * inner(y[i], z[i].dx(0)) * dx(3) - dt * h * inner(y[i], z[i].dx(0)) * dx(4)
            m0 = [[m0_0, 0, 0], [0, 0, 0], [0, 0, 0]]
            m1 = [[m1_0, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(m0),BlockForm(m1))
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
            f1_0 = zeros(Nt, dtype=object)
            f0_0[0] = inner(y_0, q[0])*dx 
            f1_0[0] = h * inner(y_0, q[0].dx(0))*dx #we don't want stabilization on the boundary
            f0 = [0, 0, f0_0] 
            f1 = [0, 0, f1_0] 
            return (BlockForm(f0), BlockForm(f1))
        elif term == "g":
            y_d = self.y_d
            z = self.z
            h = self.h
            g0_0 = zeros(Nt, dtype=object)
            g1_0 = zeros(Nt, dtype=object)
            for i in range(Nt):
                g0_0[i] = dt * y_d * z[i] * dx(3) + dt* y_d * z[i] * dx(4)
                g1_0[i] = - dt * h * y_d * z[i].dx(0) * dx(3) - dt* h * y_d * z[i].dx(0) * dx(4)
            g0 = [g0_0, 0, 0]
            g1 = [g1_0, 0, 0]
            return (BlockForm(g0),BlockForm(g1))
        elif term == "h":
            y_d = self.y_d
            h0 = y_d * y_d * dx(3, domain=mesh) + y_d * y_d * dx(4, domain=mesh) 
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = BlockDirichletBC([[[DirichletBC(self.V.sub(i), Constant(0.0), self.boundaries, 1),
                                      DirichletBC(self.V.sub(i), Constant(1.0), self.boundaries, 2),
                                      DirichletBC(self.V.sub(i), Constant(1.0), self.boundaries, 4),
                                      DirichletBC(self.V.sub(i), Constant(0.0), self.boundaries, 5),
                                      DirichletBC(self.V.sub(i), Constant(0.0), self.boundaries, 6)] for i in range(0, Nt)], None, None])
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = BlockDirichletBC([None, None, [[DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 1),
                   DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 2),
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

######MAIN PROGRAM

# Mesh
mesh = Mesh("data/graetzOC_h_0.038.xml")
subdomains = MeshFunction("size_t", mesh, "data/graetzOC_h_0.038_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/graetzOC_h_0.038_facet_region.xml")
print("hMax: ", mesh.hmax() )

# Create Finite Element space (Lagrange P1)
T = 3.0 #5.0
dt = 0.1 #1/6.0
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


# Allocate an object of the EllipticOptimalControl class
elliptic_optimal_control = EllipticOptimalControl(block_V, subdomains=subdomains, boundaries=boundaries, T=T, dt=dt, Nt=Nt)
mu_range =  [(1e4, 1e6)]
elliptic_optimal_control.set_mu_range(mu_range)
beta_a = [2 for _ in range(1)]
beta_b = [9 for _ in range(1)]


offline_mu = (100000,)
elliptic_optimal_control.init()
elliptic_optimal_control.set_mu(offline_mu)
elliptic_optimal_control.solve()
elliptic_optimal_control.export_solution(filename="FEM_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_STAB_mu_1e5_alpha_0.01")


# ### Prepare reduction with a reduced basis method

# In[ ]:


pod_galerkin_method = PODGalerkin(elliptic_optimal_control)
pod_galerkin_method.set_Nmax(20)




# ### Perform the offline phase

# In[ ]:


#lifting_mu = (1e5,)
#elliptic_optimal_control.set_mu(lifting_mu)
pod_galerkin_method.initialize_training_set(100, sampling=BetaDistribution(beta_a, beta_b), enable_import=True) #, weight=BetaWeight(beta_a, beta_b))
reduced_elliptic_optimal_control = pod_galerkin_method.offline()


# ### Perform an online solve

# In[ ]:


online_mu = (100000,)


reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_solution = reduced_elliptic_optimal_control.solve(online_stabilization=False) 
print("NOT ONLINE STAB: Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())
reduced_elliptic_optimal_control.export_solution(filename="online_solution_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_OffSTAB_mu_1e5_alpha_0.01")
reduced_elliptic_optimal_control.export_error(filename="online_error_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_OffSTAB_mu_1e5_alpha_0.01")

reduced_solution = reduced_elliptic_optimal_control.solve(online_stabilization=True) 
print("ONLINE STAB: Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())
reduced_elliptic_optimal_control.export_solution(filename="online_solution_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_OffONSTAB_mu_1e5_alpha_0.01")
reduced_elliptic_optimal_control.export_error(filename="online_error_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_OffONSTAB_mu_1e5_alpha_0.01")



# ### Perform an error analysis

# In[ ]:


pod_galerkin_method.initialize_testing_set(100, sampling=BetaDistribution(beta_a, beta_b), enable_import=True)

print("\n----------------------------------------OFFLINE STABILIZATION ERROR ANALYSIS BEGINS-------------------------------------------------\n")

pod_galerkin_method.error_analysis(online_stabilization=False, filename="error_analysis_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_OffSTAB_mu_1e5_alpha_0.01")

print("\n--------------------------------------ONLINE-OFFLINE STABILIZATION ERROR ANALYSIS BEGINS--------------------------------------------\n")

pod_galerkin_method.error_analysis(online_stabilization=True, filename="error_analysis_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_OffONSTAB_mu_1e5_alpha_0.01")



# ### Perform a speedup analysis

# In[ ]:

print("\n-----------------------------------------OFFLINE STABILIZATION SPEED-UP ANALYSIS BEGINS----------------------------------------------\n")
print("")
pod_galerkin_method.speedup_analysis(online_stabilization=False, filename="speedup_analysis_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_OffSTAB_mu_1e5_alpha_0.01")
print("\n---------------------------------------ONLINE-OFFLINE STABILIZATION SPEED-UP ANALYSIS BEGINS------------------------------------------\n")
pod_galerkin_method.speedup_analysis(online_stabilization=True, filename="speedup_analysis_NOTUQ_Beta0209_Par_OCGraetz_h_0.038_OffONSTAB_mu_1e5_alpha_0.01")


