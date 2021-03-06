import numpy as np
import numpy.linalg as nla
from math import *

make_edge = lambda i,j: (i,j) if i<j else (j,i)

norm = nla.norm
pi = np.pi
# norm = lambda x: sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2])

def normalize(v):
    n = nla.norm(v)
    assert n>1e-8,"Norm of v={} is {} too close to zero (<1e-8)!!!".format(v,n)
    return v/n

equal = lambda x,y: np.isclose(nla.norm(x-y),0)


def arc_to_nodes(C,u,v,r,phi):

    C,u,v = np.array(C),np.array(u),np.array(v)

    R = r/np.cos(phi*0.25)

    n1 = C + r*(                 u                     )
    n2 = C + R*(np.cos(phi*0.25)*u + np.sin(phi*0.25)*v)
    n3 = C + R*(np.cos(phi*0.75)*u + np.sin(phi*0.75)*v)
    n4 = C + r*(     np.cos(phi)*u +      np.sin(phi)*v)

    return n1,n2,n3,n4


def nodes_to_arc(n1,n2,n3,n4):
    v = normalize(n2-n1)
    w = normalize(n3-n2)
    u = normalize(np.cross(v,np.cross(v,w)))
    phi = 2.0*np.arccos(np.dot(v,w))
    r = nla.norm(n2-n1)/np.tan(phi*0.25)
    C = n1 - r*u
    return C,u,v,r,phi

def arc_points(arc,N=100):
    C,u,v,r,phi = arc
    return [C + r*(np.cos(t)*u + np.sin(t)*v) for t in np.linspace(0,phi,N)]

def check_unit(v):
    assert np.isclose(nla.norm(v),1.0), "v={} is not a unit vector".format(v)

def check_perp(u,v):
    assert np.isclose(np.dot(u,v),0.0), "u={} and v={} are not perpendicular vectors".format(u,v)
