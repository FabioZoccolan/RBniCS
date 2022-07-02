import numpy as np
from rbnics.sampling.distributions import ausiliaryFunction as au



def TensorProductRule(d, n, rule, bounds, param=[]):  

    bounds = np.array(bounds)
    print(rule,param)
    tmpnodes, tmpweights = au.univariate_rule(int(n[0]), rule, alpha=param[0][1], beta=param[0][0])
    for j in range(1,d):
        tmp1, tmp2 = au.univariate_rule(int(n[j]), rule, alpha=param[j][1], beta=param[j][0])
                        
        tmpnodes = au.combvec(tmpnodes, tmp1)
        tmpweights = au.combvec(tmpweights, tmp2)
    
    nodes = np.zeros((d,tmpnodes.shape[1]))
    for j in range(d):
        for k in range(tmpnodes.shape[1]):
            nodes[j,k] = (bounds[j,1]-bounds[j,0])*tmpnodes[j,k] + bounds[j,0]
            #tmpweights[j,k] = (bounds[j,1]-bounds[j,0])**(2-param[0][0]-param[0][1])*tmpweights[j,k] 
            tmpweights[j,k] = (bounds[j,1]-bounds[j,0])*tmpweights[j,k]    
                
    weights = np.array([np.prod(tmpweights, axis=0)])
    count = nodes.shape[1]
    weights = np.ndarray.tolist(weights[0,:])
    
    return nodes, weights, count
