import numpy as np
from rbnics.sampling.distributions import ausiliaryFunction as au
import matplotlib.pyplot as plt

def SmolyakRule(d, q, rule, bounds, param=[], is_loguniform=False):  
    
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
    print("\n\n\nOrdine univariato", q)
    bounds = np.array(bounds)
    nodes = np.zeros((d,1))
    weights = np.zeros((1,1))
    print("d=", d)
    #print("rule=", rule)
    print("bounds=", bounds)
    #print("param=", param)
    print('\nrange =', range(max(d,q-d+1), q+1))
    for l in range(max(d,q-d+1), q+1):  
        print("l=", l)
        # l==d means that I've only alpha = (1,1,1,...,1)
        # so I cannot generate with d_tuple algorithm

        if l == d:

            print("\nIn case 1 of SmolyakRule")
            print("is_loguniform=", is_loguniform)
            tmpnodes = np.zeros((d,1))
            tmpweights = np.ones((1,1))
            for i in range(d):
                tmp1, tmp2 = au.univariate_rule(1, rule, alpha=param[i][1], beta=param[i][0])
                #Want something like nodes, weights= au.univariate_rule(n=1, rule="UniformClenshawCurtisBeta", bounds, param)
                
                if is_loguniform == False:
                    print("Case1, is_loguniform=False::begin")
                    tmpnodes[i,0] = (bounds[i,1]-bounds[i,0])*tmp1[0,0] + bounds[i,0]
                    # i don't want a vector of all weights but only the product
                    tmpweights[0,0] = (bounds[i,1]-bounds[i,0])*tmp2[0,0]*tmpweights[0,0]
                    print("Case1, is_loguniform=False::end")

                elif is_loguniform == True:
                    print("is_loguniform=True and do nothing: nodes, weights already properly scaled")                    
                    #print("tmp1, tmp2=", tmp1, tmp2)
                    #slope = (bounds[i,1]-bounds[i,0])/(1-1e-4)
                    #intercept = bounds[i,0] - slope*1e-4
                    #print("slope, intercept=", slope, intercept)
                    #tmpnodes[i,0] = slope*tmp1[i,0] + intercept
                    #tmpweights[0,0] = slope*tmp2[0,0]*tmpweights[0,0]
                    #print("tmpnodes and tmpnodes[i]=", tmpnodes, tmpnodes[i])
                    #print("tmpweights and tmpweights[i]=", tmpweights, tmpweights[i])
                    #print("succesfully calculated shifted nodes and weights")
                
            tmpweights[0,0] = (-1)**(q-l)*au.binom_coeff(d-1, q-l)*tmpweights[0,0]
            nodes = np.concatenate((nodes, tmpnodes), axis=1) # axis = 1 adds as a column
            weights = np.concatenate((weights, tmpweights), axis=1)  
            #print("nodes before=", nodes)
            #print("weights before=", weights)
            print("Ended case 1 of Smolyakrule")
            
        else:

            print("In case 2 of SmolyakRule")
            print("is_loguniform=", is_loguniform)
            ind, m = au.d_tuples(d,l)
            for i in range(m):
                gamma = ind[i,:] # we take one alpha at a time
                print("\nMaking a tensor product with alpha=", gamma, ". Note that |alpha|_1=", sum(gamma))
                #Due to the transformation of interval chosen in ausiliary_function, we have to swap a and b
                tmpnodes, tmpweights = au.univariate_rule(int(gamma[0]), rule, alpha=param[0][1], beta=param[0][0])         
                #tmpnodes, tmpweights = au.univariate_rule(1, rule, alpha=param[0][0], beta=param[0][1])
                for j in range(1,d): # here the nodes and weights are defined for integration on the standard interval
                    tmp1, tmp2 = au.univariate_rule(int(gamma[j]), rule, alpha=param[j][1], beta=param[j][0])
                    #tmp1, tmp2 = au.univariate_rule(1, rule, alpha=param[j][0], beta=param[j][1])
                    tmpnodes = au.combvec(tmpnodes, tmp1)
                    tmpweights = au.combvec(tmpweights, tmp2)
                
                for j in range(d): # here the nodes and weights are modified for integration on the custom interval specified in bounds
                    for k in range(tmpnodes.shape[1]):
     
                        if is_loguniform == False:
                            tmpnodes[j,k] = (bounds[j,1]-bounds[j,0])*tmpnodes[j,k] + bounds[j,0]
                            tmpweights[j,k] = (bounds[j,1]-bounds[j,0])*tmpweights[j,k]    
                        elif is_loguniform == True: 
                            if k ==0 : print("is_loguniform=True and do nothing: nodes, weight already properly scaled")                            
                            #slope = (bounds[j,1]-bounds[j,0])/(1-1e-4)
                            #intercept = bounds[j,0] - slope*1e-4
                            #tmpnodes[j,k] = slope*tmpnodes[j,k] + intercept
                            #tmpweights[j,k] = slope*tmpweights[j,k]
                                                       

                # product of different component of the weight associated to the same node
                tmpweights = (-1)**(q-l)*au.binom_coeff(d-1, q-l)*np.array([np.prod(tmpweights, axis=0)])
                
                nodes = np.concatenate((nodes, tmpnodes), axis=1)
                weights = np.concatenate((weights, tmpweights), axis=1) 
                #print("nodes before=", nodes)
                #print("weights before=", weights)
            print("Ended case 2 of Smolyakrule")
                
    nodes = nodes[:,1:]
    weights = np.array([weights[0,1:]])
    #print("nodes intermediate=", nodes)
    #print("weights intermediate=", weights)

    count = 1
    if d != q:    
        print("q!=d...checking for repetitions in nodes")            
        nodes, indices, count = au.no_repetitions(nodes, 1)
        weights = au.weight_compress(weights, indices, count)
    #print("nodes after=", nodes)
    #print("weights after=", weights)
    #print("count=", count)
 
    return nodes, weights, count

