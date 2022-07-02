import numpy as np
import math as math
from scipy import special
from scipy.special import beta as bt
import quadpy as quadpy
#import cs as Cs

def d_tuples(d, q):
        
    k = np.ones((1,d))
    khat = (q-d+1)*k
    ind = np.zeros((1,d))
    
    p = 0
    while k[0,d-1] <= q:
        k[0,p] = k[0,p]+1
        if k[0,p] > khat[0,p]:
            if p != d-1:
                k[0,p] = 1
                p = p+1
        #    else: ###CHECK
        #        break
        else:
            for j in range(p):
                khat[0,j] = khat[0,p]-k[0,p]+1
            k[0,0] = khat[0,0]
            p = 0
            ind = np.concatenate((ind,k))
    
    n = ind.shape[0]
    ind = ind[1:n,:]
    n = ind.shape[0]
        
    return ind, n

def combvec(m, v): # m matrix, v row vector
    
    n1 = m.shape[0]
    a = np.zeros((n1+1,m.shape[1]*v.shape[1]))
    for j in range(n1):
        for i in range(m.shape[1]):
            for k in range(v.shape[1]):
                a[j,k*m.shape[1]+i] = m[j,i]
    for i in range(m.shape[1]):
        for j in range(v.shape[1]):
            a[n1,j*m.shape[1]+i] = v[0,j]        
    
    return a    

def ClenshawCurtis_set(n):
    print("In clenshaw_curtis_set")
    if ( n == 1 ):

        x = np.zeros ( n )
        w = np.zeros ( n )
        w[0] = 2.0

    else:

        theta = np.zeros ( n )
        for i in range ( 0, n ):
          theta[i] = float ( n - 1 - i ) * np.pi / float ( n - 1 )

        x = np.cos ( theta )
        w = np.zeros ( n )

        for i in range ( 0, n ):

          w[i] = 1.0

          jhi = ( ( n - 1 ) // 2 )

          for j in range ( 0, jhi ):

            if ( 2 * ( j + 1 ) == ( n - 1 ) ):
              b = 1.0
            else:
              b = 2.0

            w[i] = w[i] - b * np.cos ( 2.0 * float ( j + 1 ) * theta[i] ) \
                 / float ( 4 * j * ( j + 2 ) + 3 )

        w[0] = w[0] / float ( n - 1 )
        for i in range ( 1, n - 1 ):
          w[i] = 2.0 * w[i] / float ( n - 1 )
        w[n-1] = w[n-1] / float ( n - 1 )
    x = np.array([x])
    w = np.array([w])
    print("Ended ClenshawCurtis_set")
    return x, w

    
def univariate_rule(n, rule, alpha=0., beta=0.): # univariate rule for interval [0,1] 

    if rule == 'Linspace':
        weights = np.ones((1,n))
        if n == 1:
            nodes = np.array([[0.5]])
        else:
            nodes = np.array([np.linspace(0,1,n)])
            weights = (1./n)*weights
    elif rule == 'ClenshawCurtis':
        nodes, weights = ClenshawCurtis_set(n) 
        nodes = nodes*0.5 + 0.5
        weights = weights*0.5
    elif rule == 'ClenshawCurtis_nested':
        if n == 1:
            nodes, weights = ClenshawCurtis_set(1)
        else:
            nodes, weights = ClenshawCurtis_set(int(2**(n-1)+1)) 
        nodes = nodes*0.5 + 0.5
        weights = weights*0.5
    elif rule == 'GaussLegendre':
        print("In GaussLegendre univariate rule")
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = np.array([nodes])*0.5 + 0.5
        weights = np.array([weights])*0.5
    elif rule == 'GaussJacobi':   
        print("In GaussJacobi univariate rule")
        #print("alpha is:",alpha,"while beta:",beta)
        nodes, weights = special.roots_jacobi(n, alpha.item()-1, beta.item()-1)
        nodes = np.array([nodes])*0.5 + 0.5
        weights = np.array([weights])/(bt(alpha.item(), beta.item()))*(1/2)**(alpha.item()+beta.item()-1)
        #print("nodes, weights=", nodes, weights)
        #print("type(nodes)=", type(nodes))
        #QUESTION: WHAT ABOUT THE FACTOR 0.5???
    elif rule == 'GaussLogUniform':
        print("In GaussLogUniform univariate rule")
        ### If we would use nodes and weights for quadrature on [1e-4,1] (and transform them in SmolyakRule/TensorProductGrid, use this code:
        ## Compute the slope and intercept of the line through (1e-4, a) and (1,b), where [a,b] is the parameter box
        #slope = (alpha.item()-beta.item())/(1-1e-4) #recall that a=beta.item() and b=alpha.item()
        #intercept = beta.item()-slope*1e-4
        ## The weight function needed is the LogUniform(a,b) density composed with the previous linear transformation
        #weight_function = lambda x: 1/( (slope*x+intercept)*(np.log(alpha.item())-np.log(beta.item())))
        ## use MIT's quadpy library to generate nodes and weights for quadrature on (1e-4,1) with weight_function as weight function       
        #moments = quadpy.tools.integrate(
        #    lambda x: [x**k*weight_function(x) for k in range(2*n)], 1e-4, 1
        #    ) #Compute the first n moments of our weight function
        #alpha, beta = quadpy.tools.chebyshev(moments) #Compute the recurrence coefficients
        #nodes, weights = quadpy.tools.scheme_from_rc(alpha, beta, mode='numpy') 

        ### If instead we directly generate the nodes and weights for [a,b]=[beta.item(), alpha.item()]
        ## use MIT's quadpy library to generate nodes and weights for quadrature on (a,b) with LogUniform(a,b) as weight function
        print("dividing factor=", np.log(alpha.item())-np.log(beta.item()))
        print("as alpha.item(), beta.item()=", alpha.item(), beta.item())
        moments = quadpy.tools.integrate(
            lambda x: [x**(k-1)/( np.log(alpha.item())-np.log(beta.item()) ) for k in range(2*n)], beta.item(), alpha.item()
            ) #Compute the first n moments of our weight function 1/-x(log(b)-log(a))
        print("moments=", moments)
        alpha, beta = quadpy.tools.chebyshev(moments) #Compute the recurrence coefficients
        print('alpha and beta precalculated')
        print('alpha,beta=', alpha, beta)
        nodes, weights = quadpy.tools.scheme_from_rc(alpha, beta, mode='numpy') 
        print("nodes and weights precalculated")
        nodes = np.array([nodes])
        print(np.isnan(nodes).any())
        weights = np.array([weights])
        #print(np.isnan(weights).any())
        print("univariate size=", n)
        print("nodes, weights=", nodes, weights)
        print("Univariate rule computed the nodes and weigths")
        


    elif rule == "UniformClenshawCurtisBeta":
        nodes, weights = ClenshawCurtis_set(n+1)
        nodes, weights = transform_by_scaling(-1, 1, bounds, nodes, weights, beta, param)

    elif rule == "UniformClenshawCurtisUniform":
        nodes, weights = ClenshawCurtis_set(n+1)
        nodes, weights = transform_by_scaling(-1, 1, bounds, nodes, weights, uniform)

    elif rule == "UniformClenshawCurtisLogUniform":
        nodes, weights = ClenshawCurtis_set(n+1)
        nodes, weights = transform_by_scaling(-1, 1, bounds, nodes, weights, loguniform)

    return nodes, weights

def beta_density(beta_a, beta_b, a, b):
    return lambda x: (1/(b-a))**(beta_a+beta_b-1)*(x-a)**(beta_a-1)*(b-x)**(beta_b-1)/bt(beta_a, beta_b)

def loguniform_density(a, b):
    return lambda x: 1/(x*(np.log(b)-np.log(a)))

def uniform_density(a, b):
    return lambda x:1/(b-a)

def shift(from_left, from_right, to_left, to_right): # from [left, right] to [bounds[0], bounds[1]]
    slope = (to_right-to_left)/(from_right-from_left)
    intercept = to_left-slope*from_left
    return lambda x: slope*x+intercept 

def inverse_shift(to_left, to_right, from_left, from_right):
    slope = (right-left)/(bounds[1]-bounds[0])
    intercept = left - slope*bounds[0]
    return lambda x: slope*x+intercept

def transform_by_scaling(bounds, nodes, weights, distribution):
    nodes = shift(-1, 1, to_left, to_right)(nodes2)
    weights= beta_density(betas[0][0], betas[0][1], box[0][0], box[0][1])(nodes_shifted)*(box[0][1]-box[0][0])/2 * weights2


def binom_coeff(n, k):
    
    return math.factorial(n)/(math.factorial(n-k)*math.factorial(k))

def no_repetitions(A, axis): # axis can be 0 or 1
    
    if axis == 1:
        n = A.shape[1]
        m = A.shape[0]
        A_reduced = np.zeros((m,1))
        A_reduced[:,0] = A[:,0]
        n_reduced = 1
        count = [[0]]
        if n>1:
            for k in range(1,n):
                control = 1
                for i in range(n_reduced):
                    if np.array_equal(A_reduced[:,i],A[:,k]):
                        print("repetition found in column", k)
                        count[i].append(k)
                        control = 0
                if control:
                    A_reduced = np.concatenate((A_reduced,np.reshape(A[:,k],(m,1))),1)
                    n_reduced = n_reduced + 1
                    count.append([k])
    else:
        A_reduced, count, n_reduced = no_repetitions(A.T,1)
        A_reduced = A_reduced.T
    
    return A_reduced, count, n_reduced
    
def weight_compress(v, indices, count):

    w = np.zeros((1, count))
    for i in range(count):
        for j in range(len(indices[i])):
            w[0,i] = w[0,i]+v[0,int(indices[i][j])]
    w = np.ndarray.tolist(w)[0]   
                
    return w                

def array2tuple(m, axis): # axis can be 0 or 1
    
    if axis == 1:
        v = []
        for j in range(m.shape[1]):
            v.append(tuple(np.ndarray.tolist(m[:,j])))
    else:
        v = array2tuple(m.T, 1) 
    
    return v

