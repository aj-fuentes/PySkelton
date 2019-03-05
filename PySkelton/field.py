# -*- coding: UTF-8 -*-
import math
import numpy as np
import numpy.linalg as nla

from . import skeleton as sk
from . import nformulas as nf
import pyroots as pr

_default_radii = np.ones(2,dtype=float)
_default_angles = np.zeros(2,dtype=float)

_pr_brent = pr.Brentq(epsilon=1e-6)
_brent = lambda g,a,b: _pr_brent(g,a,b).x0
# _brent = sco.brentq

class Field(object):
    """docstring for Field"""

    def __init__(self, R, skel, a, b, c, th, gsl_ws_size=100, max_error=1.0e-8):

        assert isinstance(skel,sk.Skeleton),"<skel> must be an instace of Skeleton class, current type is: {}".format(type(skel))

        skel.field = self #connect the skeleton with this field

        self.R = float(R)
        self.skel = skel
        self.gsl_ws_size = int(gsl_ws_size)
        self.max_error = float(max_error)

        self.l = skel.l
        #compute maximum distance from the skeleton for field eval
        self.max_r = max(max(a),max(b),max(c))

        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.th = np.array(th)

    def eval(self, X):
        raise NotImplementedError()

    def newton_eval(self, Q, m, s):
        raise NotImplementedError()

    def gradient_eval(self, X):
        raise NotImplementedError()

    def parametric_gradient_eval(self, X):
        raise NotImplementedError()

    def hessian_eval(self, X):
        raise NotImplementedError()

    def shoot_ray(self, Q, m, level_value, guess_R=None, tol=1e-7, max_iters=100):

        R = self.R
        g = lambda s: self.eval(Q+m*s)-level_value

        assert g(0.0)>0.0,"Q={} is not an interior point of the surface".format(Q)


        def find_negative_point(a,b,N=10):
            for x in np.linspace(a,b,N,endpoint=False):
                if g(x)<0.0:
                    return x
            return None

        #find the initial interval
        iters = max_iters
        low,upp = 0.0,R
        x = find_negative_point(low,upp)
        while x is None:
            low,upp = upp,upp+R
            x = find_negative_point(low,upp)
            iters -= 1
            if iters==0: raise ValueError("No intersection for point Q={} and vector m={} for maximum radius R={}".format(Q,m,R))

        #take as upper limit the negative point
        upp = x
        #with the right interval compute the root
        try:
            upp = _brent(g,0.0,upp)
        except pr.utils.ConvergenceError as e:
            if e.args[0]!="Bracket is smaller than tolerance.":
                raise pr.utils.ConvergenceError(e.args) from e
            else:
                upp *= 0.5

        iters = max_iters
        x = find_negative_point(0.0,upp)
        while not (x is None):
            # print "Searching for new root closer than {}".format(x)
            try:
                upp = _brent(g,0.0,x)
            except pr.utils.ConvergenceError as e:
                if e.args[0]!="Bracket is smaller than tolerance.":
                    raise pr.utils.ConvergenceError(e.args) from e
                else:
                    upp = 0.5*x
                    break

            x = find_negative_point(low,upp)
            iters-=1
            if iters==0: break

        return Q+m*upp

    def shoot_ray_old(self, Q, m, level_value, guess_R=None, tol=1e-7, max_iters=100):

        f = self.eval
        R = self.R
        # if guess_R:
        #     R = guess_R
        a = 0.0
        b = R

        assert f(Q)>level_value,"Q={} is not an interior point of the surface".format(Q)

        iters = max_iters
        while f(Q+m*b)>level_value:
            if iters>0:
                iters -= 1
                a,b=b,b+R
            else:
                raise ValueError("No intersection for point Q={} and vector m={} for maximum radius R={}".format(Q,m,b))



        g = lambda s: f(Q+m*s)-level_value
        # s0 = sco.brentq(g,a,b)
        s0 = _brent(g,a,b)

        ##BELOW an attempt to find the closest intersection
        # while True:
        #     try:
        #         d = 0.001*(s0-a)
        #         s0 = sco.brentq(g,a,s0-d)
        #     except:
        #         break

        return Q+m*s0

class SegmentField(Field):

    def __init__(self, R, segment, a=_default_radii, b=_default_radii, c=_default_radii, th=_default_angles, gsl_ws_size=100, max_error=1.0e-8):
        super(SegmentField,self).__init__(R,segment,a,b,c,th,gsl_ws_size,max_error)

        assert isinstance(segment,sk.Segment),"<segment> must be an instace of skeleton.Segment class"

        self.P = segment.P
        self.T = segment.v
        self.N = segment.get_normal_at(0)

    def eval(self, X):
        return nf.compact_field_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,self.gsl_ws_size,self.max_error)

    def gradient_eval(self,X):
        g0 = nf.compact_gradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,0,self.gsl_ws_size,self.max_error)
        g1 = nf.compact_gradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,1,self.gsl_ws_size,self.max_error)
        g2 = nf.compact_gradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,2,self.gsl_ws_size,self.max_error)
        return np.array([g0,g1,g2])

    # def parametric_gradient_eval(self, X, vals=[0,1,2,3,4,5,6,7]):
    #     da0 = nf.compact_pgradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,0,self.gsl_ws_size,self.max_error)
    #     da1 = nf.compact_pgradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,1,self.gsl_ws_size,self.max_error)
    #     db0 = nf.compact_pgradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,2,self.gsl_ws_size,self.max_error)
    #     db1 = nf.compact_pgradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,3,self.gsl_ws_size,self.max_error)
    #     dc0 = nf.compact_pgradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,4,self.gsl_ws_size,self.max_error)
    #     dc1 = nf.compact_pgradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,5,self.gsl_ws_size,self.max_error)
    #     d_0 = nf.compact_pgradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,6,self.gsl_ws_size,self.max_error)
    #     d_1 = nf.compact_pgradient_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.th,self.max_r,self.R,7,self.gsl_ws_size,self.max_error)
    #     res = np.array([da0,da1,db0,db1,dc0,dc1,d_0,d_1])
    #     return res[vals]

class ArcField(Field):

    def __init__(self, R, arc, a=_default_radii, b=_default_radii, c=_default_radii, th=_default_angles, gsl_ws_size=100, max_error=1.0e-8):
        super(ArcField,self).__init__(R,arc,a,b,c,th,gsl_ws_size,max_error)

        assert isinstance(arc,sk.Arc),"<arc> must be an instace of skeleton.Arc class, current type is: {}".format(type(arc))

        self.C = arc.C
        self.u = arc.u
        self.v = arc.v
        self.r = arc.r
        self.phi = arc.phi

    def eval(self, X):
        return nf.arc_compact_field_eval(X,self.C,self.r,self.u,self.v,self.phi,self.a,self.b,self.c,self.th,self.max_r,self.R,self.gsl_ws_size,self.max_error)

    def gradient_eval(self, X):
        g0 = nf.arc_compact_gradient_eval(X,self.C,self.r,self.u,self.v,self.phi,self.a,self.b,self.c,self.th,self.max_r,self.R,0,self.gsl_ws_size,self.max_error)
        g1 = nf.arc_compact_gradient_eval(X,self.C,self.r,self.u,self.v,self.phi,self.a,self.b,self.c,self.th,self.max_r,self.R,1,self.gsl_ws_size,self.max_error)
        g2 = nf.arc_compact_gradient_eval(X,self.C,self.r,self.u,self.v,self.phi,self.a,self.b,self.c,self.th,self.max_r,self.R,2,self.gsl_ws_size,self.max_error)
        return np.array([g0,g1,g2])

class MultiField(Field):

    def __init__(self, fields):

        self.fields = fields
        self.R = max(f.R for f in fields)
        self.coeffs = [1.0]*len(fields)

    def eval(self,X):
        return sum(c*f.eval(X) for c,f in zip(self.coeffs,self.fields))

    def gradient_eval(self, X):
        return sum(c*f.gradient_eval(X) for c,f in zip(self.coeffs,self.fields))

    def set_coeff(self,i,c):
        self.coeffs[i] = c

    def set_coeffs(self,cs):
        assert len(cs)==len(self.fields),"Number of coeffs must be equal to the number of fields"
        self.coeffs = cs

class G1Field(Field):

    def __init__(self, R, curve, a=_default_radii, b=_default_radii, c=_default_radii, th=_default_angles, gsl_ws_size=100, max_error=1.0e-8):
        super(G1Field,self).__init__(R,curve,a,b,c,th,gsl_ws_size,max_error)

        assert isinstance(curve,sk.G1Curve),"<curve> must be an instance of skeleton.G1Curve class, current type is: {}".format(type(curve))

        def convex_combination(xs,a,b,T):
            return np.array([(xs[0]*(T-a) + xs[1]*a)/T,(xs[0]*(T-b) + xs[1]*b)/T])

        L = curve.l
        self.fields = []
        # print("total a={} b={} c={} th={}".format(self.a,self.b,self.c,self.th))
        for angle,l,skel in zip(curve.angles,curve.ls,curve.skels):
            l1 = l-skel.l
            l2 = l
            a = convex_combination(self.a,l1,l2,L)
            b = convex_combination(self.b,l1,l2,L)
            c = convex_combination(self.c,l1,l2,L)
            th2 = convex_combination(self.th+angle,l1,l2,L)
            # print("piece a={} b={} c={} th={}".format(a,b,c,th2))
            if isinstance(skel,sk.Segment):
                self.fields.append(SegmentField(R,skel,a,b,c,th2))
            elif isinstance(skel,sk.Arc):
                self.fields.append(ArcField(R,skel,a,b,c,th2))

    def eval(self,X):
        return sum(f.eval(X) for f in self.fields)

    def gradient_eval(self, X):
        return sum(f.gradient_eval(X) for f in self.fields)

def make_field(R, skel, a=_default_radii, b=_default_radii, c=_default_radii, th=_default_angles, gsl_ws_size=100, max_error=1.0e-8):
    assert isinstance(skel,sk.Skeleton),"<skel> is not an instance of PySkelton.Skeleton"
    klass = None
    if isinstance(skel,sk.Segment):
        klass = SegmentField
    elif isinstance(skel,sk.Arc):
        klass = ArcField
    elif isinstance(skel,sk.G1Curve):
        klass = G1Field
    return klass(R, skel, a, b, c, th, gsl_ws_size, max_error)

def get_eigenval_param(r,R,level_set):
    # term = math.pow(0.5*level_set,2.0/7.0)
    eta = get_eta_constant(level_set)
    eigenval = ((R/r)*eta)**2
    return eigenval

def get_omega_constant(level_set):
    # = 0.5493568319351
    p = lambda x: x-(x**3)+3.0/5.0*(x**5)-(x**7)/7.0 - (1-level_set)*16.0/35.0
    _pr_brent2 = pr.Brentq(epsilon=1e-10)
    _brent2 = lambda g,a,b: _pr_brent(g,a,b).x0
    return _brent2(p,0,1)

def get_eta_constant(level_set):
    res = math.sqrt(1-math.pow(level_set*0.5,2/7))
    return res

def get_tangential_eigenval_param(r,R,level_set):
    w = get_omega_constant(level_set)
    eigenval = ((R/r)*w)**2
    return eigenval

def get_radius_param(r,R,level_set):
    eigenval = get_eigenval_param(r,R,level_set)
    radius = 1.0/math.sqrt(eigenval)
    return radius

def get_tangential_radius_param(r,R,level_set):
    eigenval = get_tangential_eigenval_param(r,R,level_set)
    radius = 1.0/math.sqrt(eigenval)
    return radius
