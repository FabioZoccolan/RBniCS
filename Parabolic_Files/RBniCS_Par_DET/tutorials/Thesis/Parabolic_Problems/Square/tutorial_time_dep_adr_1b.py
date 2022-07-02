#
# RBniCS and multiphenics are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and multiphenics are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from multiphenics import *
from rbnics import *
from numpy import ceil, isclose, zeros


@PullBackFormsToReferenceDomain()
@ShapeParametrization(
    ("x[0]", "x[1]"), # subdomain 1
    ("mu[1]*(x[0] - 1) + 1", "x[1]"), # subdomain 2
    ("mu[1]*(x[0] - 1) + 1", "x[1]"), # subdomain 3
)

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
        # Store the velocity expression
        self.vel = Expression("x[1]*(1-x[1])", degree=2, domain=mesh)
        # Store the initial condition at t = 0
        self.y_0 = Expression(("1*(x[0] == 0)"), degree=2, domain=mesh)
        # Customize linear solver parameters
        self._linear_solver_parameters.update({
            "linear_solver": "umfpack"
        })
    # Return custom problem name
    def name(self):
        return "TimeDependentADROptimalControl1BPOD"
        
    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term in ("a", "a*"):
            theta_a0 = 1.0/mu[0]
            theta_a1 = 1.0/(mu[0]*mu[1])
            theta_a2 = mu[1]/mu[0]
            theta_a3 = 1.0
            return (theta_a0, theta_a1, theta_a2, theta_a3)
        elif term in ("c", "c*"):
            theta_c0 = mu[1]
            return (theta_c0,)
        elif term == "m":
            theta_m0 = mu[1]
            return (theta_m0,)
        elif term == "n":
            theta_n0 = self.alpha*mu[1]
            return (theta_n0,)
        elif term == "f":
            theta_f0 = 1.0
            return (theta_f0,)
        elif term == "g":
            theta_g0 = mu[1]*mu[2]
            return (theta_g0,)
        elif term == "h":
            theta_h0 = 0.4*mu[1]*mu[2]**2
            return (theta_h0,)
        elif term == "dirichlet_bc_y":
            theta_bc0 = 1.
            return (theta_bc0,)
        else:
            raise ValueError("Invalid term for compute_theta().")
            
    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        dx = self.dx
        ds = self.ds
        dt = self.dt
        print(term)
        if term == "a":
            y = self.y
            q = self.q
            vel = self.vel
            a0_0 = zeros((Nt, Nt), dtype=object)
            a1_0 = zeros((Nt, Nt), dtype=object)
            a2_0 = zeros((Nt, Nt), dtype=object)
            a3_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                a0_0[i, i] = dt*inner(grad(y[i]), grad(q[i]))*dx(1)
                a1_0[i, i] = dt*y[i].dx(0)*q[i].dx(0)*dx(2) + dt*y[i].dx(0)*q[i].dx(0)*dx(3)
                a2_0[i, i] = dt*y[i].dx(1)*q[i].dx(1)*dx(2) + dt*y[i].dx(1)*q[i].dx(1)*dx(3)
                a3_0[i, i] = inner(y[i],q[i])*dx + dt*vel*y[i].dx(0)*q[i]*dx
            for i in range(Nt-1):
                a3_0[i+1,i] = - inner(y[i], q[i+1])*dx
            a0 = [[0, 0, 0], [0, 0, 0], [a0_0, 0, 0]]
            a1 = [[0, 0, 0], [0, 0, 0], [a1_0, 0, 0]]
            a2 = [[0, 0, 0], [0, 0, 0], [a2_0, 0, 0]]
            a3 = [[0, 0, 0], [0, 0, 0], [a3_0, 0, 0]]
            return (BlockForm(a0), BlockForm(a1), BlockForm(a2), BlockForm(a3))
        elif term == "a*":
            z = self.z
            p = self.p
            vel = self.vel
            as0_0 = zeros((Nt, Nt), dtype=object)
            as1_0 = zeros((Nt, Nt), dtype=object)
            as2_0 = zeros((Nt, Nt), dtype=object)
            as3_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                as0_0[i, i] = dt*inner(grad(z[i]), grad(p[i]))*dx(1)
                as1_0[i, i] = dt*z[i].dx(0)*p[i].dx(0)*dx(2) + dt*z[i].dx(0)*p[i].dx(0)*dx(3)
                as2_0[i, i] = dt*z[i].dx(1)*p[i].dx(1)*dx(2) + dt*z[i].dx(1)*p[i].dx(1)*dx(3)
                as3_0[i, i] = inner(z[i],p[i])*dx - dt*vel*p[i].dx(0)*z[i]*dx
            for i in range(Nt-1):
                as3_0[i+1, i] = - inner(p[i+1], z[i])*dx
            as0 = [[0, 0, as0_0], [0, 0, 0], [0, 0, 0]]
            as1 = [[0, 0, as1_0], [0, 0, 0], [0, 0, 0]]
            as2 = [[0, 0, as2_0], [0, 0, 0], [0, 0, 0]]
            as3 = [[0, 0, as3_0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(as0), BlockForm(as1), BlockForm(as2), BlockForm(as3))
        elif term == "c":
            u = self.u
            q = self.q
            c0_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                c0_0[i,i] = + dt*inner(u[i], q[i])*ds(2)
            c0 = [[0, 0, 0], [0, 0, 0], [0, c0_0, 0]]
            return(BlockForm(c0),)
        elif term == "c*":
            v = self.v
            p = self.p
            cs0_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                cs0_0[i,i] = + dt*inner(p[i], v[i])*ds(2)
            cs0 = [[0, 0, 0], [0, 0, cs0_0], [0, 0, 0]]
            return(BlockForm(cs0),)
        elif term == "m":
            y = self.y
            z = self.z
            m0_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                m0_0[i,i] = dt*inner(y[i], z[i])*dx(3)
            m0 = [[m0_0, 0, 0], [0, 0, 0], [0, 0, 0]]
            return (BlockForm(m0),)
        elif term == "n":
            u = self.u
            v = self.v
            n0_0 = zeros((Nt, Nt), dtype=object)
            for i in range(Nt):
                n0_0[i,i] = dt*inner(u[i], v[i])*ds(2)
            n0 = [[0, 0, 0], [0, n0_0, 0], [0, 0, 0]]
            return (BlockForm(n0),)
        elif term == "f":
            q = self.q
            y_0 = self.y_0
            f0_0 = zeros(Nt, dtype=object)
            f0_0[0] = inner(y_0, q[0])*dx
            f0 = [0, 0, f0_0]
            return (BlockForm(f0),)
        elif term == "g":
            z = self.z
            g0_0 = zeros(Nt, dtype=object)
            for i in range(Nt):
                g0_0[i] = dt*z[i]*dx(3)
            g0 = [g0_0, 0, 0]
            return (BlockForm(g0),)
        elif term == "h":
            h0 = 1.0
            return (h0,)
        elif term == "dirichlet_bc_y":
            bc0 = BlockDirichletBC([[DirichletBC(block_V.sub(i), Constant(1.), self.boundaries, 1) for i in range(0, Nt)], None, None])
            return (bc0,)
        elif term == "dirichlet_bc_p":
            bc0 = BlockDirichletBC([None, None, [DirichletBC(block_V.sub(i), Constant(0.0), self.boundaries, 1) for i in range(2*Nt, 3*Nt)]])
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

# 1. Read the mesh for this problem
mesh = Mesh("data/mesh3.xml")
subdomains = MeshFunction("size_t", mesh, "data/mesh3_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/mesh3_facet_region.xml")
control_boundary = MeshRestriction(mesh, "data/mesh3_control_restriction.rtc.xml")

# 2. Create Finite Element space (Lagrange P1)
T = 1.
dt = 0.5
Nt = int(ceil(T/dt))
# BOUNDARY RESTRICTIONS #
y_restrict = []
u_restrict = []
p_restrict = []
for i in range(Nt):
    y_restrict.append(None)
    u_restrict.append(control_boundary)
    p_restrict.append(None)

# FUNCTION SPACES #
scalar_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element = BlockElement([scalar_element]*Nt + [scalar_element]*Nt + [scalar_element]*Nt)
components = ["y"]*Nt + ["u"]*Nt + ["p"]*Nt
block_V = BlockFunctionSpace(mesh, element, restrict = [*y_restrict, *u_restrict, *p_restrict], components=[*components])

# 3. Allocate an object of the EllipticOptimalControl class
elliptic_optimal_control = EllipticOptimalControl(block_V, subdomains=subdomains, boundaries=boundaries, T=T, dt=dt, Nt=Nt)
mu_range = [(6.0, 20.0), (1.0, 3.0), (0.5, 3.0)]
elliptic_optimal_control.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
pod_galerkin_method = PODGalerkin(elliptic_optimal_control)
pod_galerkin_method.set_Nmax(3)

# 5. Perform the offline phase
pod_galerkin_method.initialize_training_set(5)
reduced_elliptic_optimal_control = pod_galerkin_method.offline()

# 6. Perform an online solve
online_mu = (12.0, 2.0, 2.5)
reduced_elliptic_optimal_control.set_mu(online_mu)
reduced_elliptic_optimal_control.solve()
reduced_elliptic_optimal_control.export_solution(filename="online_solution")
elliptic_optimal_control.solve()
elliptic_optimal_control.export_solution(filename="offline_solution")
print("Reduced output for mu =", online_mu, "is", reduced_elliptic_optimal_control.compute_output())
# 7. Perform an error analysis
aaaaaaaaa
pod_galerkin_method.initialize_testing_set(100)
pod_galerkin_method.error_analysis()

# 8. Perform a speedup analysis
pod_galerkin_method.speedup_analysis()
