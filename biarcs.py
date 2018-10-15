import numpy as np
import numpy.linalg as nla
from visualization import *
import pyroots as pr
import math


_pr_brent = pr.Brentq(epsilon=1e-12)
_brent = lambda x,a,b: _pr_brent(x,a,b).x0

def positive_root(a,b,c):
    sq = math.sqrt(b*b-4.0*a*c)
    x = (-b + sq)/(2.0*a)
    if x<0.0:
        x = (-b - sq)/(2.0*a)
    assert x>0.0,"No positive root for {}x^2 + {}x + {} = 0".format(a,b,c)
    return x

def get_arc_from_points(A,B,C):
    p = A-B
    q = C-B
    assert np.isclose(nla.norm(p),nla.norm(q)),"The points {} and {} are not equidistant to {}".format(A,C,B)

    n = np.cross(p,q)
    n /= nla.norm(n)

    v = -p/nla.norm(p)

    u = np.cross(n,v)
    u /= nla.norm(u)

    w = np.cross(n,q)
    w /=nla.norm(w)

    r = nla.norm(C-A)/nla.norm(w-u)

    phi = math.atan2(np.dot(w,v),np.dot(w,u))

    C0 = A-r*u

    assert np.isclose(abs(nla.norm(A-C0)-nla.norm(C-C0)),0.0),"The center is not equidistant of the extremities: r1={}, r2={}".format(nla.norm(A-C0),nla.norm(C-C0))

    assert np.isclose(np.dot(u,p),0.0),"{} is not tangent to the circle".format(p)

    assert np.isclose(np.dot(C-C0,q),0.0),"{} is not tangent to the circle".format(q)

    assert np.isclose(nla.norm(C0 + r*math.cos(phi)*u + r*math.sin(phi)*v - C),0.0),"The endpoint of the circle does not coincide with {}".format(C)

    return (C0,u,v,r,phi)


def plot(gamma,a,b,N=100,name="curve",color=green,origin=None,vis=None):
    P = origin
    if P is None:
        P = np.array([.0,.0,.0])
    ts = np.linspace(a,b,N)
    ps = [gamma(t)+P for t in ts]
    vis.add_polyline(ps,color=color,name=name)

def compute_biarc(A0,t0,A1,t1):

    assert np.isclose(nla.norm(t0),1.0),"The norm of unit tangent t={} is not 1.0 (={})".format(t0,nla.norm(t0))
    assert np.isclose(nla.norm(t1),1.0),"The norm of unit tangent t={} is not 1.0 (={})".format(t1,nla.norm(t1))

    a = np.dot(t0+t1,t0+t1)-4.0
    b = 2.0*np.dot(t0+t1,A0-A1)
    c = np.dot(A0-A1,A0-A1)

    l = positive_root(a,b,c)

    L = A0+l*t0
    N = A1-l*t1
    M = 0.5*(L+N)
    a1 = get_arc_from_points(A0,L,M)
    a2 = get_arc_from_points(M,N,A1)

    # a1 = (C0,n1,n2,r1,2*phi1)
    # a2 = (C1,n1,n2,r2,2*phi2)

    _,u,v,_,phi = a1
    t0final = -math.sin(phi)*u + math.cos(phi)*v
    assert np.isclose(nla.norm(v-t0),0.0),"The tangent of the first circle is not the expected one"

    _,u,v,_,phi = a2
    assert np.isclose(nla.norm(t0final-v),0.0),"The two circles have no common tangent"
    assert np.isclose(nla.norm(-math.sin(phi)*u + math.cos(phi)*v - t1),0.0),"The tangent of the second circle is not the expected one"

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

def biarcs_from_Hdata(As,vs):
    aa = []
    N = len(As)
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


