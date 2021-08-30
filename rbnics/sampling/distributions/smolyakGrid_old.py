import numpy as np
from rbnics.sampling.distributions import ausiliaryFunction as au
import matplotlib.pyplot as plt

def SmolyakRule(d,q,rule,bounds,param=[]):  
    

    # d is the dimension of the integral and the quadrature nodes too.
    # q is the quadrature rule
    # a is the set of parameter
    # bounds is the vector that contains the bounds of the interval
    if len(param)==0:
        raise TypeError("No parameter setted for the distribution")
    if np.array(bounds).shape!=np.array(param).shape:
       raise TypeError("Bound shape and beta shape differ")
    if (d<=0 or q<=0):
        raise TypeError("d or q are <=0")
    print("\n\n\nOrdine univariato",q)
    bounds = np.array(bounds)
    nodes = np.zeros((d,1))
    weights = np.zeros((1,1))
    for l in range(max(d,q-d+1),q+1):
        # l==d means that I've only alpha = (1,1,1,...,1)
        # so I cannot generate with d_tuple algorithm

        if l == d:

            tmpnodes = np.zeros((d,1))
            tmpweights = np.ones((1,1))
            for i in range(d):

                tmp1, tmp2 = au.univariate_rule(1,rule,alpha=param[i][1],beta=param[i][0])
                tmpnodes[i,0] = (bounds[i,1]-bounds[i,0])*tmp1[0,0] + bounds[i,0]
                # i don't want a vector of all weights but only the product
                tmpweights[0,0] = (bounds[i,1]-bounds[i,0])*tmp2[0,0]*tmpweights[0,0]

            tmpweights[0,0] = (-1)**(q-l)*au.binom_coeff(d-1,q-l)*tmpweights[0,0]
            nodes = np.concatenate((nodes,tmpnodes),axis=1) # axis = 1 adds as a column
            weights = np.concatenate((weights,tmpweights),axis=1)
            
        else:

            ind, m = au.d_tuples(d,l)
            
            for i in range(m): 

                gamma = ind[i,:] # we take one alpha at a time
                tmpnodes, tmpweights = au.univariate_rule(int(gamma[0]),rule,alpha=param[0][1],beta=param[0][0])         
                #tmpnodes, tmpweights = au.univariate_rule(1,rule,alpha=param[0][0],beta=param[0][1])
                for j in range(1,d):
                    tmp1, tmp2 = au.univariate_rule(int(gamma[j]),rule,alpha=param[j][1],beta=param[j][0])
                    #tmp1, tmp2 = au.univariate_rule(1,rule,alpha=param[j][0],beta=param[j][1])
                    tmpnodes = au.combvec(tmpnodes,tmp1)
                    tmpweights = au.combvec(tmpweights,tmp2)
                
                for j in range(d):
                    for k in range(tmpnodes.shape[1]):
                        tmpnodes[j,k] = (bounds[j,1]-bounds[j,0])*tmpnodes[j,k] + bounds[j,0]
                        tmpweights[j,k] = (bounds[j,1]-bounds[j,0])*tmpweights[j,k]    
                
                # product of different component of the weight associated to the same node
                tmpweights = (-1)**(q-l)*au.binom_coeff(d-1,q-l)*np.array([np.prod(tmpweights, axis=0)])
                
                nodes = np.concatenate((nodes,tmpnodes),axis=1)
                weights = np.concatenate((weights,tmpweights),axis=1) 
                
               

    nodes = nodes[:,1:]
    weights = np.array([weights[0,1:]])
    
    count = 1
    if d != q:                
        nodes, indices, count = au.no_repetitions(nodes,1)
        weights = au.weight_compress(weights,indices,count)
    
 
    return nodes, weights, count

