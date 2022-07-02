#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dolfin import *
from mshr import *


# In[ ]:


# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
subdomain1 = Rectangle(Point(0.0, 0.0), Point(1.0, 0.25))
subdomain2 = Rectangle(Point(0.0, 0.25), Point(1.0, 0.75))
subdomain3 = Rectangle(Point(0.0, 0.75), Point(0.25, 1.0))
subdomain4 = Rectangle(Point(0.25, 0.75), Point(1.0, 1.0))
domain.set_subdomain(1, subdomain1)  # add some fake subdomains to make sure that the mesh is split
domain.set_subdomain(2, subdomain2)  # at x[1] = 0.25, since boundary id changes at (0, 0.25)
domain.set_subdomain(3, subdomain3) 
domain.set_subdomain(4, subdomain4) 
mesh = generate_mesh(domain, 27)
plot(mesh)

# In[ ]:


# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
#subdomains.set_all(0)
plot(subdomains)

# In[ ]:


# Create boundaries
class Boundary1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            abs(x[1] - 0.) < DOLFIN_EPS
            or (abs(x[0] - 0.) < DOLFIN_EPS and x[1] <= 0.25)
        )


class Boundary2(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            abs(x[1] - 1.) < DOLFIN_EPS
            or abs(x[0] - 1.) < DOLFIN_EPS
            or (abs(x[0] - 0.) < DOLFIN_EPS and x[1] >= 0.25)
        )



boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
boundary_1 = Boundary1()
boundary_1.mark(boundaries, 1)
boundary_2 = Boundary2()
boundary_2.mark(boundaries, 2)


# In[ ]:


# Save
File("squareOC.xml") << mesh
File("squareOC_physical_region.xml") << subdomains
File("squareOC_facet_region.xml") << boundaries
XDMFFile("squareOC.xdmf").write(mesh)
XDMFFile("squareOC_physical_region.xdmf").write(subdomains)
XDMFFile("squareOC_facet_region.xdmf").write(boundaries)

