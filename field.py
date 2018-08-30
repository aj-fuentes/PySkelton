from skeleton import Skeleton
import scipy.optimize as sco
import nformulas as nf


class Field(object):
    """docstring for Field"""

    def __init__(self, R, skel, gsl_ws_size=100, max_error=1.0e-8):

        assert isinstance(skel,Skeleton),"<skel> must be an instace of Skeleton class"

        self.R = float(R)
        self.skel = skel
        self.gsl_ws_size = int(gsl_ws_size)
        self.max_error = float(max_error)

        self.l = skel.l
        self.a = skel.a
        self.b = skel.b
        self.c = skel.c


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

        iters = max_iters
        while f(Q+m*b)>level_value:
            if iters>0:
                iters -= 1
                a,b=b,b+R
            else:
                raise ValueError('No intersection for point Q=%s and vector m=%s for maximum radius R=%s' % (Q,m,b))


        g = lambda s: f(Q+m*s)-level_value
        s0 = sco.brentq(g,a,b)

        ##BELOW an attempt to find the closest intersection
        # while True:
        #     try:
        #         d = 0.001*(s0-a)
        #         s0 = sco.brentq(g,a,s0-d)
        #     except:
        #         break

        return Q+m*s0

class SegmentField(Field):

    def __init__(self, R, segment, gsl_ws_size=100, max_error=1.0e-8):
        super(SegmentField,self).__init__(R,segment,gsl_ws_size,max_error)

        self.P = segment.P
        self.T = segment.v
        self.N = segment.get_normal_at(0)

    def eval(self, X):
        return nf.compact_field_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.R,self.gsl_ws_size,self.max_error)


class ArcField(Field):

    def __init__(self, R, arc, gsl_ws_size=100, max_error=1.0e-8):
        super(ArcField,self).__init__(R,arc,gsl_ws_size,max_error)

        self.C = arc.C
        self.u = arc.u
        self.v = arc.v
        self.r = arc.r
        self.phi = arc.phi

    def eval(self, X):
        return nf.arc_compact_field_eval(X,self.C,self.r,self.u,self.v,self.phi,self.a,self.b,self.c,self.R,self.gsl_ws_size,self.max_error)


class MultiField(Field):

    def __init__(self, fields):

        self.fields = fields
        self.R = max(f.R for f in fields)

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

