import numpy as np
from rbnics.sampling.distributions import ausiliaryFunction as au



def TensorProductRule(d, n, rule, bounds, param=[], is_loguniform=False):  
    print("In TensorProductRule")
    bounds = np.array(bounds)
    print(rule, param)
    print("bounds= ", bounds)
    print("n=", n)
    print("d=", d)
    tmpnodes, tmpweights = au.univariate_rule(int(n[0]), rule, alpha=param[0][1], beta=param[0][0])
    for j in range(1,d):
        tmp1, tmp2 = au.univariate_rule(int(n[j]), rule, alpha=param[j][1], beta=param[j][0])
                        
        tmpnodes = au.combvec(tmpnodes, tmp1)
        tmpweights = au.combvec(tmpweights, tmp2)
    
    nodes = np.zeros((d,tmpnodes.shape[1]))
    print("tmpnodes.shape[1]=", tmpnodes.shape[1])
    print("nodes before=", nodes)
    for j in range(d):
        for k in range(tmpnodes.shape[1]):
            
            if is_loguniform == False:
                print("is_loguniform=", is_loguniform)
                nodes[j,k] = (bounds[j,1]-bounds[j,0])*tmpnodes[j,k] + bounds[j,0]
                tmpweights[j,k] = (bounds[j,1]-bounds[j,0])*tmpweights[j,k]  
                #tmpweights[j,k] = (bounds[j,1]-bounds[j,0])**(2-param[0][0]-param[0][1])*tmpweights[j,k] 
            elif is_loguniform == True:
                print("is_loguniform=", is_loguniform)
                #The line between (1e-4,1) and (bounds[j,0],bounds[j,1])=:(a,b) has slope (b-a)(1-1e-4) and intercept a-slope*1e-4
                slope = (bounds[j,1]-bounds[j,0])/(1-1e-4)
                intercept = bounds[j,0] - slope*1e-4
                nodes[j,k] = slope*tmpnodes[j,k] + intercept
                tmpweights[j,k] = slope*tmpweights[j,k]

    print("nodes after=", nodes)
    weights = np.array([np.prod(tmpweights, axis=0)])
    count = nodes.shape[1]
    print("count=", count)
    weights = np.ndarray.tolist(weights[0,:])
    print("weights=", weights)
    
    return nodes, weights, count
