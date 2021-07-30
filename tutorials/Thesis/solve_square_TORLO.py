# Copyright (C) 2015-2016 SISSA mathLab
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#
## @file solve_square.py
#  @brief Example 4: Square test case
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from RBniCS import *
import time
import numpy as np
import matplotlib.pyplot as plt



#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 4: square CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class Square(EllipticCoerciveRBBase):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, V, mesh, subd, bound):
        # Store the BC object for the homogeneous solution (after lifting)
        bc_list = [
            DirichletBC(V, 0.0, bound, 1), # indeed homog. bcs
            DirichletBC(V, 0.0, bound, 2), # indeed homog. bcs
            DirichletBC(V, 0.0, bound, 3), # indeed homog. bcs
            DirichletBC(V, 0.0, bound, 4), # non-homog. bcs with a lifting
            DirichletBC(V, 0.0, bound, 5)  # non-homog. bcs with a lifting
        ]
        # Call the standard initialization
        super(Square, self).__init__(V, bc_list)
        # ... and also store FEniCS data structures for assembly
        self.dx = Measure("dx")(subdomain_data=subd)
        self.ds = Measure("ds")(subdomain_data=bound)
        # We will consider non-homogeneous Dirichlet BCs with a lifting.
        # First of all, assemble a suitable lifting function
        lifting_bc = [ 
            DirichletBC(V, 1.0, bound, 1), # homog. bcs
            DirichletBC(V, 0.0, bound, 2), # homog. bcs
            DirichletBC(V, 0.0, bound, 3), # homog. bcs
            DirichletBC(V, 0.0, bound, 4), # non-homog. bcs
            DirichletBC(V, 1.0, bound, 5)  # non-homog. bcs
        ]
        u = self.u
        v = self.v
        dx = self.dx
        lifting_a = inner(grad(u),grad(v))*dx
        lifting_A = assemble(lifting_a)
        lifting_f = 1e-15*v*dx
        lifting_F = assemble(lifting_f)
        [bc.apply(lifting_A) for bc in lifting_bc] # Apply BCs on LHS
        [bc.apply(lifting_F) for bc in lifting_bc] # Apply BCs on RHS
        lifting = Function(V)
        solve(lifting_A, lifting.vector(), lifting_F)
        # Discard the lifting_{bc, A, F} object and store only the lifting function
        self.lifting = lifting
        self.export_basis(self.lifting, self.basis_folder + "lifting")
        # Store the velocity expression
        self.vel = Expression("1.", degree=self.V.ufl_element().degree())
        # Finally, initialize an SCM object to approximate alpha LB
        self.SCM_obj = SCM_Square(self)
        
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     SETTERS     ########################### 
    ## @defgroup Setters Set properties of the reduced order approximation
    #  @{
    
    # Propagate the values of all setters also to the SCM object
    
    def setNmax(self, nmax):
        EllipticCoerciveRBBase.setNmax(self, nmax)
        self.SCM_obj.setNmax(2*nmax)
    def settol(self, tol):
        EllipticCoerciveRBBase.settol(self, tol)
        self.SCM_obj.settol(tol)
    def setmu_range(self, mu_range):
        EllipticCoerciveRBBase.setmu_range(self, mu_range)
        self.SCM_obj.setmu_range(mu_range)
    def setxi_train(self, ntrain, enable_import=False, sampling="random"):
        EllipticCoerciveRBBase.setxi_train(self, ntrain, enable_import, sampling)
        self.SCM_obj.setxi_train(ntrain, True, sampling)
    def setxi_test(self, ntest, enable_import=False, sampling="random"):
        EllipticCoerciveRBBase.setxi_test(self, ntest, enable_import, sampling)
        self.SCM_obj.setxi_test(ntest, True, sampling)
    def setmu(self, mu):
        EllipticCoerciveRBBase.setmu(self, mu)
        self.SCM_obj.setmu(mu)

    def set_weighted_flag(self, weighted_flag):
        EllipticCoerciveRBBase.set_weighted_flag(self, weighted_flag)
        self.SCM_obj.set_weighted_flag(weighted_flag)

    def set_density(self, weight):
        EllipticCoerciveRBBase.set_density(self, weight)
        self.SCM_obj.set_density(weight)

        
    #  @}
    ########################### end - SETTERS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Return the alpha_lower bound.
    def get_alpha_lb(self):
        return self.SCM_obj.get_alpha_LB(self.mu)
    
    ## Set theta multiplicative terms of the affine expansion of a.
    def compute_theta_a(self):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        theta_a0 = 1./mu1
        theta_a1 = np.cos(mu2)
        theta_a2 = np.sin(mu2)
        if stabilization==True:
            theta_a3 = self.delta * np.cos(mu2)**2
            theta_a4 = self.delta * np.sin(mu2)*np.cos(mu2)
            theta_a5 = self.delta * np.sin(mu2)**2
        else:
            theta_a3 = 0.
            theta_a4 = 0.
            theta_a5 = 0.

        return (theta_a0, theta_a1, theta_a2, theta_a3, theta_a4, theta_a5)
    
    ## Set theta multiplicative terms of the affine expansion of f.
    def compute_theta_f(self):
        mu1 = self.mu[0]
        mu2 = self.mu[1]
        theta_f0 = - 1./mu1
        theta_f1 = - np.cos(mu2)
        theta_f2 = - np.sin(mu2)
        if stabilization==True:
            theta_f3 = - self.delta * np.cos(mu2)**2
            theta_f4 = - self.delta * np.sin(mu2)*np.cos(mu2)
            theta_f5 = - self.delta * np.sin(mu2)**2
        else:
            theta_f3 = 0.
            theta_f4 = 0.
            theta_f5 = 0.
        return (theta_f0, theta_f1, theta_f2, theta_f3, theta_f4, theta_f5)
        
    
    ## Set matrices resulting from the truth discretization of a.
    def assemble_truth_a(self):
        u = self.u
        v = self.v
        dx = self.dx
        vel = self.vel
        h = self.h
        # Define
        a0 = inner(grad(u),grad(v))*dx + 1e-15*u*v*dx
        a1 = vel*u.dx(0)*v*dx + 1e-15*u*v*dx
        a2 = vel*u.dx(1)*v*dx + 1e-15*u*v*dx
        a3 = h*vel*u.dx(0)*v.dx(0)*dx + 1e-15*u*v*dx
        a4 = h*vel*(u.dx(0)*v.dx(1)+u.dx(1)*v.dx(0))*dx + 1e-15*u*v*dx
        a5 = h*vel*u.dx(1)*v.dx(1)*dx + 1e-15*u*v*dx
        # Assemble
        A0 = assemble(a0)
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)
        A4 = assemble(a4)
        A5 = assemble(a5)
        # Return
        return (A0, A1, A2, A3, A4, A5)
    
    ## Set vectors resulting from the truth discretization of f.
    def assemble_truth_f(self):
        v = self.v
        dx = self.dx
        vel = self.vel
        lifting = self.lifting
        h=self.h
        # Define
        f0 = inner(grad(lifting),grad(v))*dx + 1e-15*lifting*v*dx
        f1 = vel*lifting.dx(0)*v*dx + 1e-15*lifting*v*dx
        f2 = vel*lifting.dx(1)*v*dx + 1e-15*lifting*v*dx
        f3 = h*vel*lifting.dx(0)*v.dx(0)*dx + 1e-15*lifting*v*dx
        f4 = h*vel*(lifting.dx(0)*v.dx(1)+lifting.dx(1)*v.dx(0))*dx + 1e-15*lifting*v*dx
        f5 = h*vel*lifting.dx(1)*v.dx(1)*dx + 1e-15*lifting*v*dx
        # Assemble
        F0 = assemble(f0)
        F1 = assemble(f1)
        F2 = assemble(f2)
        F3 = assemble(f3)
        F4 = assemble(f4)
        F5 = assemble(f5)
        # Return
        return (F0, F1, F2, F3, F4, F5)
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 
    
    ###########################     OFFLINE STAGE     ########################### 
    ## @defgroup OfflineStage Methods related to the offline stage
    #  @{
    
    ## Perform the offline phase of the reduced order model
    def offline(self):
        # Perform first the SCM offline phase, ...
        bak_first_mu = tuple(list(self.mu))
        self.SCM_obj.offline()
        # ..., and then call the parent method.
        self.setmu(bak_first_mu)
        EllipticCoerciveRBBase.offline(self)
    
    #  @}
    ########################### end - OFFLINE STAGE - end ########################### 
    
    ###########################     I/O     ########################### 
    ## @defgroup IO Input/output methods
    #  @{
    
    ## Preprocess the solution before plotting to add a lifting
    def preprocess_solution_for_plot(self, solution):
        solution_with_lifting = Function(self.V)
        solution_with_lifting.vector()[:] = solution.vector()[:] + self.lifting.vector()[:]
        return solution_with_lifting
        
    #  @}
    ########################### end - I/O - end ########################### 
    
    ###########################     ERROR ANALYSIS     ########################### 
    ## @defgroup ErrorAnalysis Error analysis
    #  @{
    
    # Compute the error of the reduced order approximation with respect to the full order one
    # over the test set
    def error_analysis(self, N=None):
        # Perform first the SCM error analysis, ...
        self.SCM_obj.error_analysis()
        # ..., and then call the parent method.
        EllipticCoerciveRBBase.error_analysis(self, N)        
        
    #  @}
    ########################### end - ERROR ANALYSIS - end ########################### 
    
#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 4: SCM CLASS     ~~~~~~~~~~~~~~~~~~~~~~~~~# 
class SCM_Square(SCM):
    
    ###########################     CONSTRUCTORS     ########################### 
    ## @defgroup Constructors Methods related to the construction of the reduced order model object
    #  @{
    
    ## Default initialization of members
    def __init__(self, parametrized_problem):
        # Call the standard initialization
        SCM.__init__(self, parametrized_problem)
        
        # Good guesses to help convergence of bounding box
        self.guess_bounding_box_minimum = (1.e-5, 1.e-6, 1.e-5, 1.e-5, -0.15, 1.e-5)
        self.guess_bounding_box_maximum = (100., 7., 1., 1., 1., 1.)
    
    #  @}
    ########################### end - CONSTRUCTORS - end ########################### 
    
    ###########################     PROBLEM SPECIFIC     ########################### 
    ## @defgroup ProblemSpecific Problem specific methods
    #  @{
    
    ## Set additional options for the eigensolver (bounding box minimum)
    def set_additional_eigensolver_options_for_bounding_box_minimum(self, eigensolver, qa):
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["spectral_shift"] = self.guess_bounding_box_minimum[qa]
        
    ## Set additional options for the eigensolver (bounding box maximimum)
    def set_additional_eigensolver_options_for_bounding_box_maximum(self, eigensolver, qa):
        eigensolver.parameters["spectral_transform"] = "shift-and-invert"
        eigensolver.parameters["spectral_shift"] = self.guess_bounding_box_maximum[qa]
        
    #  @}
    ########################### end - PROBLEM SPECIFIC - end ########################### 


#~~~~~~~~~~~~~~~~~~~~~~~~~     EXAMPLE 4: MAIN PROGRAM     ~~~~~~~~~~~~~~~~~~~~~~~~~# 

# 1. Read the mesh for this problem
mesh = Mesh("data/square2.xml")
subd = MeshFunction("size_t", mesh, "data/square2_physical_region.xml")
bound = MeshFunction("size_t", mesh, "data/square2_facet_region.xml")
hmax=mesh.hmax()
hmin=mesh.hmin()
h=CellSize(mesh)

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)
print( V.dofmap().dofs() )

# 3. Allocate an object of the Square class
square = Square(V, mesh, subd, bound)
square.h=h

# 4. Stabilizazione online/offline
stab_online=True
stab_offline=True
square.delta=1.

# 4. Choose PETSc solvers as linear algebra backend
parameters.linear_algebra_backend = 'PETSc'

# 5. Define a Beta distribution over parameters for greedy algorithm and set mu range
mu_range = [(1.e4, 1.e5), (0., 1.5)] #(-0.75, 1.57)]

square.setmu_range(mu_range)


alfabeta=[(3.,4.),(4.,2.)]
distribution1 = BetaDistribution(alpha=alfabeta)
distribution = UniformDistribution()

density = BetaWeight(alpha=alfabeta)
density1 = UniformWeight()
# original_density = NameWeight()
# density = IndicatorWeight(original_density, 0.1)


square.setxi_train(100,sampling=distribution)
square.setNmax(20)
square.set_density(weight=density1)
square.set_weighted_flag(1)
stabilization=stab_offline

square.setmu((1.e4, 0.7))

time_init= time.time()
square.offline()
print "TRUTHn TIME = ", time.time()-time_init

square.set_density(weight=density)

stabilization=stab_online
time_init= time.time()
approx=square.online_solve()
print "RB TIME = ", time.time()-time_init
exit() #DELETE ME
#square.export_solution(approx, "plot/RB_solution_mu="+ str(square.mu)+"_beta=40")

#square.setxi_test(100,  sampling=distribution1) #enable_import=True,
#square.error_analysis()

#error=square.error_u
#delta=square.delta_u

#np.save("error_u_random", error)
#np.save("delta_u_random", delta)


square.setxi_test(200, sampling=distribution1)
xi_test_weight = density.density(square.mu_range, square.xi_test)


k=0
sum_weight=0.
sum_err_stab=0.
sum_err_instab=0.

errori_stab=[]
peclet=[]
errori_instab=[]
stop_stab=[]
stop_instab=[]
stop=[]

for j in range(len(square.xi_test)):
    online_mu=square.xi_test[j]
    peclet += [online_mu[1]]
    weight=xi_test_weight[j] 
    print "mu = ",online_mu, ", weight = ", weight
    square.setmu(online_mu)
    

    stabilization=stab_offline
    square.truth_A = square.assemble_truth_a()
    square.apply_bc_to_matrix_expansion(square.truth_A)
    square.truth_F = square.assemble_truth_f()
    square.apply_bc_to_vector_expansion(square.truth_F)
    square.Qa = len(square.truth_A)
    square.Qf = len(square.truth_F)

    truth=square.truth_solve()
    
    stabilization=True        
    square.truth_A = square.assemble_truth_a()
    square.apply_bc_to_matrix_expansion(square.truth_A)
    square.truth_F = square.assemble_truth_f()
    square.apply_bc_to_vector_expansion(square.truth_F)
    square.Qa = len(square.truth_A)
    square.Qf = len(square.truth_F)

    approx=square.online_solve(with_plot=False)


    errore=Function(V)
    errore.vector()[:]=truth.vector() - approx.vector() # error
    
    square.theta_a = square.compute_theta_a() # not really necessary, for symmetry with the parabolic case
    assembled_truth_A_sym = square.affine_assemble_truth_symmetric_part_matrix(square.truth_A, square.theta_a)
    error_norm_squared = square.compute_scalar(errore, errore, assembled_truth_A_sym) # norm of the error

    errrr=np.sqrt(error_norm_squared)

    errori_stab += [errrr]
    err_weighted=errrr*weight
    sum_err_stab += err_weighted

    stabilization=False        

    square.truth_A = square.assemble_truth_a()
    square.apply_bc_to_matrix_expansion(square.truth_A)
    square.truth_F = square.assemble_truth_f()
    square.apply_bc_to_vector_expansion(square.truth_F)
    square.Qa = len(square.truth_A)
    square.Qf = len(square.truth_F)

    approx=square.online_solve(with_plot=False)

    # 9. Compute energy norm error
    errore=Function(V)
    errore.vector()[:]=truth.vector() - approx.vector() # error
    
    square.theta_a = square.compute_theta_a() # not really necessary, for symmetry with the parabolic case
    assembled_truth_A_sym = square.affine_assemble_truth_symmetric_part_matrix(square.truth_A, square.theta_a)
    error_norm_squared = square.compute_scalar(errore, errore, assembled_truth_A_sym) # norm of the error
    errrr=np.sqrt(error_norm_squared)

    errori_instab += [errrr]
    err_weighted=errrr*weight
    sum_err_instab += err_weighted
    sum_weight +=weight


err_stab=sum_err_stab/sum_weight
err_instab=sum_err_instab/sum_weight
np.save("integrale_random", err_stab)

rrrr
weight_sorted=sorted(xi_test_weight)

tol=sum_weight*0.1
tmp=0.
j=0
#while tmp < tol:
#    j+=1
#    tmp+=weight_sorted[j]
#weight_discr=weight_sorted[j]

#print "Avra' funzionato?"
#print weight_discr
#print j
#print len(square.xi_test)

err=0.
cont=0
for j in range(len(square.xi_test)):
    if square.xi_test[j][1]<2.:
        err+=errori_instab[j]*xi_test_weight[j]
        cont+=1
    else:
        err+=errori_stab[j]*xi_test_weight[j]
err_discriminant=np.sqrt(err/sum_weight)
    
print cont
print err_stab
print err_instab
print err_discriminant


