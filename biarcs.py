import numpy as np
import numpy.linalg as nla
from visualization import *
import pyroots as pr
import math


_pr_brent = pr.Brentq(epsilon=1e-8)
_brent = lambda x,a,b: _pr_brent(x,a,b).x0

def plot(gamma,a,b,N=100,name="curve",color=green,origin=None,vis=None):
    P = origin
    if P is None:
        P = np.array([.0,.0,.0])
    ts = np.linspace(a,b,N)
    ps = [gamma(t)+P for t in ts]
    vis.add_polyline(ps,color=color,name=name)

def compute_biarc(A0,t0,A1,t1):
    v = A1-A0

    a = np.dot(v,v)
    b = 2*np.dot(v,t0)
    c = 2*np.dot(v,t1)
    d = 2*(np.dot(t0,t1)-1)
    f = lambda x: a - b*x - c*x + d*x*x
    l0 = l1 = _brent(f,0.,10.0,)

    L = A0+t0*l0
    N = A1-t1*l1
    p = N-L
    M = L+(l0/(l0+l1))*p

    #first arc
    u0 = (l0+l1)/(l0+l1-np.dot(p,t0))
    LC0 = u0*(p*l0/(l0+l1)-l0*t0)
    C0 = L+LC0

    C0A0 = A0-C0
    r1 = nla.norm(C0A0)
    phi1 = np.arccos(np.dot(C0A0,-LC0)/nla.norm(-LC0)/r1)
    n2 = t0
    n1 = C0A0/r1
    a1 = (C0,n1,n2,r1,2*phi1)

    #second arc
    u1 = (l0+l1)/(l0+l1-np.dot(p,t1))
    NC1 = u1*(-p*l1/(l0+l1)+l1*t1)
    C1 = N+NC1

    C1M = M-C1
    r2 = nla.norm(C1M)
    phi2 = np.arccos(np.dot(C1M,-NC1)/nla.norm(-NC1)/r2)
    n2 = p/nla.norm(p)
    n1 = C1M/r2
    a2 = (C1,n1,n2,r2,2*phi2)

    return [a1,a2]

def biarcs_from_curve(gamma,gammat,a,b,N):
    ts = np.linspace(a,b,N)
    As = [gamma(t) for t in ts]
    vs = [gammat(t) for t in ts]

    aa = []
    for i in range(N-1):
        A0,A1 = As[i],As[i+1]
        t0,t1 = vs[i],vs[i+1]
        aa.extend(compute_biarc(A0,t0,A1,t1))
    return aa

def plot_biarcs(aa,origin=None,vis=None):
    O = origin
    if O is None:
        O = np.array([.0,.0,.0])
    for P,n1,n2,r,phi in aa:
        arc1 = lambda t: P+n1*np.cos(t)*r+n2*math.sin(t)*r
        plot(arc1,0,phi,name="arcs",color=magenta,origin=O,vis=vis)


if __name__=="__main__":


    def helix(t,c,d,e):
        return np.array([c*np.cos(t),d*math.sin(t),e*t])
    def helixt(t,c,d,e):
        v = np.array([-c*math.sin(t),d*np.cos(t),e])
        v /= nla.norm(v)
        return v
    def helixn(t,c,d,e):
        v = np.array([-c*math.sin(t),d*np.cos(t),e])
        w = np.array([-c*np.cos(t),-d*math.sin(t),0.0])
        u = np.cross(v,np.cross(w,v))
        u /= nla.norm(u)
        return u


    gamma = lambda t: helix(t,2,3,1.5)
    gammat = lambda t: helixt(t,2,3,1.5)
    gamman = lambda t: helixn(t,2,3,1.5)

    a = 0
    K = 2
    b = K*np.pi

    vis = Visualization2()

    O = np.array([.0,.0,.0])
    plot(gamma,a,b,20*K,origin=O,vis=vis)
    aa = biarcs_from_curve(gamma,gammat,a,b,4)
    plot_biarcs(aa,O,vis=vis)

    vis.show()


