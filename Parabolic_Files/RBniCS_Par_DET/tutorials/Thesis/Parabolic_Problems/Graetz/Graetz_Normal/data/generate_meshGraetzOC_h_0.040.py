#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dolfin import *
from mshr import *


# In[ ]:


# Create mesh
domain = Rectangle(Point(0., 0.), Point(2., 1.))
#subdomain = dict()
subdomain1 = Rectangle(Point(0., 0.), Point(1., 1.))
subdomain2 = Rectangle(Point(1., 0.2), Point(2., 0.8))
subdomain3 = Rectangle(Point(1., 0.), Point(2., 0.2))
subdomain4 = Rectangle(Point(1., 0.8), Point(2., 1.))
domain.set_subdomain(1, subdomain1)  
domain.set_subdomain(2, subdomain2) 
domain.set_subdomain(3, subdomain3) 
domain.set_subdomain(4, subdomain4) 
mesh = generate_mesh(domain, 50) #50 --> 9360
plot(mesh)


# In[ ]:


# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())
plot(subdomains)


# In[ ]:


# Create boundaries
class Left(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 0.) < DOLFIN_EPS


class Right(SubDomain):
    def __init__(self):
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 2.) < DOLFIN_EPS


class Bottom(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 0.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max


class Top(SubDomain):
    def __init__(self, x_min, x_max):
        SubDomain.__init__(self)
        self.x_min = x_min
        self.x_max = x_max

    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS and x[0] >= self.x_min and x[0] <= self.x_max


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
bottomLeft = Bottom(0., 1.)
bottomLeft.mark(boundaries, 1)
bottomRight = Bottom(1., 2.)
bottomRight.mark(boundaries, 2)
right = Right()
right.mark(boundaries, 3)
topRight = Top(1, 2.)
topRight.mark(boundaries, 4)
topLeft = Top(0., 1.)
topLeft.mark(boundaries, 5)
left = Left()
left.mark(boundaries, 6)


# In[ ]:


# Save
File("graetzOC_h_0.041.xml") << mesh
File("graetzOC_h_0.041_physical_region.xml") << subdomains
File("graetzOC_h_0.041_facet_region.xml") << boundaries
XDMFFile("graetzOC_h_0.041.xdmf").write(mesh)
XDMFFile("graetzOC_h_0.041_physical_region.xdmf").write(subdomains)
XDMFFile("graetzOC_h_0.041_facet_region.xdmf").write(boundaries)
