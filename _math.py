import numpy as np
import numpy.linalg as nla

norm = nla.norm

def normalize(v):
    n = nla.norm(v)
    assert n>1e-8,"Norm of v={} is {} too close to zero (<1e-8)!!!".format(v,n)
    return v/n

equal = lambda x,y: np.isclose(nla.norm(x-y),0)
