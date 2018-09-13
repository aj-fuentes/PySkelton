import skeleton as sk
# import scipy.optimize as sco
import nformulas as nf
import pyroots as pr
import numpy as np

_default_radii = np.ones(2,dtype=float)
_default_angles = np.zeros(2,dtype=float)

_pr_brent = pr.Brentq()
_brent = lambda x,a,b: _pr_brent(x,a,b).x0
# _brent = sco.brentq

class Field(object):
    """docstring for Field"""

    def __init__(self, R, skel, a, b, c, th, gsl_ws_size=100, max_error=1.0e-8):

        assert isinstance(skel,sk.Skeleton),"<skel> must be an instace of Skeleton class, current type is: {}".format(type(skel))

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

    def hessian_eval(self, X):
        raise NotImplementedError()

    def shoot_ray(self, Q, m, level_value, tol=1e-7, max_iters=100):

        f = self.eval
        R = self.R
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

class MultiField(Field):

    def __init__(self, fields):

        self.fields = fields
        self.R = max(f.R for f in fields)

    def eval(self,X):
        return sum(f.eval(X) for f in self.fields)

class G1Field(Field):

    def __init__(self, R, curve, a=_default_radii, b=_default_radii, c=_default_radii, th=_default_angles, gsl_ws_size=100, max_error=1.0e-8):
        super(G1Field,self).__init__(R,curve,a,b,c,th,gsl_ws_size,max_error)

        assert isinstance(curve,sk.G1Curve),"<curve> must be an instance of skeleton.G1Curve class, current type is: {}".format(type(curve))

        def convex_combination(xs,a,b,T):
            return np.array([(xs[0]*(T-a) + xs[1]*a)/T,(xs[0]*(T-b) + xs[1]*b)/T])

        L = curve.l
        self.fields = []
        for angle,l,skel in zip(curve.angles,curve.ls,curve.skels):
            l2 = l+skel.l
            a = convex_combination(a,l,l2,L)
            b = convex_combination(b,l,l2,L)
            c = convex_combination(c,l,l2,L)
            th2 = convex_combination(th+angle,l,l2,L)
            if isinstance(skel,sk.Segment):
                self.fields.append(SegmentField(R,skel,a,b,c,th2))
            elif isinstance(skel,sk.Arc):
                self.fields.append(ArcField(R,skel,a,b,c,th2))

    def eval(self,X):
        return sum(f.eval(X) for f in self.fields)
