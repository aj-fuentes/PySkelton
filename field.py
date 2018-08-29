import numpy as np
import numpy.linalg as nal
import scipy.optimize as sco
from nformulas import *


class Field(object):
    """docstring for Field"""

    def __init__(self, R, skel, a, b, c, gsl_ws_size=100, max_error=1.0e-8):
        super(Field, self).__init__()
        self.R = R
        self.skel = skel
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.n = gsl_ws_size
        self.e = max_error


    def eval(self, X):
        raise NotImplementedError()

    def newton_eval(self, Q, m, s):
        raise NotImplementedError()

    def gradient_eval(self, X):
        raise NotImplementedError()

    def hessian_eval(self, X):
        raise NotImplementedError()

    def shoot_ray(self, Q, m, level_value, tol=1e-7, max_iters=100, initial_dist=None, double_distance=0, newton=True):

        f = self.eval
        R = self.R
        if not (initial_dist is None):
            R = initial_dist

        a = 0.0
        b = R

        iters = 2**double_distance
        while f(Q+m*b)>level_value:
            if iters>0:
                iters -= 1
                a,b=b,b+R
            else:
                raise ValueError('No intersection for point Q=%s and vector m=%s for maximum radius R=%s' % (Q,m,b))

        # newton=False
        # tol = 1e-7
        if newton:
            g = lambda s: f(Q+m*s)-level_value
            #gp = lambda s: self.newton_eval(Q,m,s)
            #s0 = sco.newton(g,0.0,gp)
            s0 = sco.brentq(g,a,b)

            ##BELOW an attempt to find the closest intersection
            # while True:
            #     try:
            #         d = 0.001*(s0-a)
            #         s0 = sco.brentq(g,a,s0-d)
            #     except:
            #         break
            return Q+m*s0
        else:
            T = Q+m*(a+b)/2.0
            fval = f(T)

            while abs(fval-level_value)>tol and max_iters>0:
                if fval>level_value:
                    b = (a+b)/2.0
                else:
                    a = (a+b)/2.0
                T = Q+m*(a+b)/2.0
                fval = f(T)
                max_iters -= 1

            if max_iters<=0:
                raise ValueError('Is not possible to find the intersection point within tolerance tol=%s, Q=%s and vector m=%s' % (tol,Q,m))
            else:
                return T

class SegmentField(Field):

    def __init__(self, R, segment, normal, a, b, c, gsl_ws_size=100, max_error=1.0e-8):
        super(SegmentField,self).__init__(R,segment,a,b,c,gsl_ws_size,max_error)

        self.P = segment.P
        self.T = segment.n
        self.N = normal
        self.l = segment.l

        if not np.isclose(np.dot(self.T,self.N),0.0):
            raise ValueError("Tangent {} and normal {} are not perpendicular".format(self.T,self.N))

    def eval(self, X):
        return compact_field_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.R,self.n,self.e)


class ArcField(Field):

    def __init__(self, R, arc, a, b, c, gsl_ws_size=100, max_error=1.0e-8):
        super(ArcField,self).__init__(R,arc,a,b,c,gsl_ws_size,max_error)

        self.C = arc.P
        self.u = arc.n1
        self.v = arc.n2
        self.r = arc.r
        self.phi = arc.phi

        if not np.isclose(np.dot(self.u,self.v),0.0):
            raise ValueError("Arc axes {} and {} are not perpendicular".format(self.u,self.v))

    def eval(self, X):
        #print arc_compact_field_eval(X,np.array([0.0,0.0,0.0],5.0,))
        return arc_compact_field_eval(X,self.C,self.r,self.u,self.v,self.phi,self.a,self.b,self.c,self.R,self.n,self.e)


class MultiField(Field):

    def __init__(self, *fields):
        super(MultiField, self).__init__(max([f.R for f in fields]),None)

        self.fields = fields

    def __getitem__(self, i):
        return self.fields[i]

    def __iter__(self):
        return self.fields.__iter__()

    def eval(self,X):
        return sum(f.eval(X) for f in self.fields)

    def newton_eval(self, Q, m, s):
        return sum(f.newton_eval(Q, m, s) for f in self.fields)

    def gradient_eval(self, Q):
        return sum(f.gradient_eval(Q) for f in self.fields)

