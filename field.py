import numpy as np
import numpy.linalg as nal
import scipy.optimize as sco
from nformulas.nformulas import *


class Field(object):

    standard_metric = np.array([
            [ 1.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],
            [ 0.0, 0.0, 1.0]
            ])

    """docstring for Field"""
    def __init__(self, R, skel, metric1=None, metric2=None,power=False):
        super(Field, self).__init__()
        self.R = R
        self.skel = skel
        self.symbolic_integral = None
        self.power = power

        w0 = 1.0 if skel is None else skel.initial_weight
        w1 = 1.0 if skel is None else skel.final_weight
        l  = 1.0 if skel is None else float(skel.get_arc_lenght())

        self.q = w0
        self.p = (w1-w0)/l

        self.metric = (not metric1 is None) or (not metric2 is None)
        self.metric = True #for now use always the metric version

        if metric1 is None:
            metric1=np.copy(self.standard_metric)
            metric1[0,0] *= w0
            metric1[1,1] *= w0
            metric1[2,2] *= w0

        if metric2 is None:
            metric2=np.copy(self.standard_metric)
            metric2[0,0] *= w1
            metric2[1,1] *= w1
            metric2[2,2] *= w1

        self.g0 = metric1
        self.g1 = metric2

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

class ArcField(Field):

    def __init__(self, R, arc,metric1=None, metric2=None,power=False):
        super(ArcField, self).__init__(R,arc,metric1,metric2,power)

        self.arc = self.skel

    def get_integral_limits(self, X):

        R = self.R

        a = self.arc
        P,r,n1,n2,r,u_max = a.P,a.r,a.n1,a.n2,a.r,a.u_max

        u0 = 0
        u1 = 0

        #solve the equation |G(t)-X| = R in terms of u
        #this means the intersection between an arc and a sphere
        A = r**2+(P-X).dot(P-X)-R**2
        B = 2*r*n1.dot(P-X)
        C = 2*r*n2.dot(P-X)

        disc = -A**2 + B**2 + C**2

        #check if there is any itnersection aka if this value is positive
        if disc >= 0 :
            D = np.sqrt(disc)
            u0 = (-C + D)/(A - B)
            u1 = -(C + D)/(A - B)

            if u0>u1:
                u0,u1 = u1,u0

            if u0 < 0: u0 = 0
            if u1 < 0: u1 = 0
            if u0 > u_max: u0 = u_max
            if u1 > u_max: u1 = u_max

        return u0,u1

    def get_symbolic_field_integral(self):

        import sympy as sy
        from sympy.parsing.sympy_parser import parse_expr
        from formulas import arc_formula

        a,b,c = self.arc.P
        u1,v1,w1 = self.arc.n1
        u2,v2,w2 = self.arc.n2
        r = self.arc.r
        R = self.R
        pformula = parse_expr(arc_formula,{
            'a':a,
            'b':b,
            'c':c,
            'u1':u1,
            'v1':v1,
            'w1':w1,
            'u2':u2,
            'v2':v2,
            'w2':w2,
            'r':r,
            'R':R,
            '_':sy.atan
        })

        return sy.lambdify(('t','x','y','z'),pformula)

        # R = self.R

        # a = self.arc
        # P,n1,n2,r = a.P,a.n1,a.n2,a.r

        # return get_arc_integral_function(P,n1,n2,R,r)

    def eval(self, Q):
        # if not self.symbolic_integral:
        #     self.symbolic_integral = self.get_symbolic_field_integral()

        # a,b = self.get_integral_limits(X)
        # return self.symbolic_integral(b,*X)-self.symbolic_integral(a,*X)

        P = self.arc.P
        n1 = self.arc.n1
        n2 = self.arc.n2
        R = self.R
        phi = self.arc.phi
        r = self.arc.r

        g0 = self.g0
        g1 = self.g1

        if self.power:
            #if the point is very close to the arc
            if(nal.norm(Q-P)<1e-8):
                return 1e10 #return a very high value
            else:
                return napi6m_a(Q,P,n1,n2,r,R,g0,g1,100,1e-8,phi)
        return nacp3m_a(Q,P,n1,n2,r,R,g0,g1,100,1e-8,phi)


class SegmentField(Field):

    def __init__(self, R, segment, metric1=None, metric2=None,power=False):
        super(SegmentField, self).__init__(R,segment,metric1,metric2,power)

        self.segment = self.skel

        # self.p = 0.0#1.0/float(self.segment.l)
        # self.q = 1.0



    def get_integral_limits(self, X):

        P = self.segment.P
        n = self.segment.n
        R = self.R
        l = self.segment.l

        s = np.dot((X-P),n)
        d = nal.norm(X-(P+s*n))
        if d<R:
            r = np.sqrt(R*R-d*d)
            low = s-r
            upp = s+r
            if low < 0: low = 0
            if upp < 0: upp = 0
            if low > l: low = l
            if upp > l: upp = l
        else:
            low = 0
            upp = 0

        return (low,upp)



    def get_symbolic_field_integral(self):

        import sympy as sy
        from sympy.parsing.sympy_parser import parse_expr
        from formulas import segment_formula

        a,b,c = self.segment.P
        u,v,w = self.segment.n
        R = self.R
        pformula = parse_expr(segment_formula,{
            'a':a,
            'b':b,
            'c':c,
            'u':u,
            'v':v,
            'w':w,
            'R':R,
            '_':sy.atan
        })

        return sy.lambdify(('t','x','y','z'),pformula)

    def eval(self,Q):

        # if not self.symbolic_integral:
        #     self.symbolic_integral = self.get_symbolic_field_integral()

        P = self.segment.P
        n = self.segment.n
        R = self.R
        l = self.segment.l

        if self.metric:
            g0 = self.g0
            g1 = self.g1
            if self.power:
                v = Q-P
                tt = n.dot(v) #parameter value for the projection
                if tt<0.0:
                    tt = 0.0
                if tt>l:
                    tt = l
                #if the point is very close to segment
                if nal.norm(Q-P-n*tt)<1e-8:
                    return 1e10 #return a very high value
                else:
                    return napi6m(Q,P,n,R,g0,g1,100,1e-8,l)
            return nacp3m(Q,P,n,R,g0,g1,100,1e-8,l)

        a = 2*np.dot(P-Q,n)
        b = np.dot(P-Q,P-Q)

        #try with another metric
        x = P-Q
        a = 2*(x.dot(n))
        b = x.dot(x)

        p = self.p
        q = self.q
        return nacp3(a,b,p,q,R,0,self.segment.l,1e-8,100)

    def newton_eval(self, Q, m, s):

        P = self.segment.P
        n = self.segment.n
        R = self.R

        a = 2*np.dot(P-Q,n)
        b = np.dot(P-Q,P-Q)

        c = -2*np.dot(P-Q,m)
        d = -2*np.dot(m,n)

        p = self.p
        q = self.q

        return nancp3(a,b,c,d,p,q,R,0,self.segment.l,s,1e-8,100)

    def gradient_eval(self, Q):
        P = self.segment.P
        n = self.segment.n
        R = self.R
        l = self.segment.l

        x = P-Q
        a = 2*x.dot(n)
        b = x.dot(x)

        p = self.p
        q = self.q

                                                           #a,b,p,q,R,Pj  ,vj  ,xj ,t0,t1,error,interval_division
        d0 = nadcp3(a,b,p,q,R,P[0],n[0],Q[0],0,l,1e-8,100)
        d1 = nadcp3(a,b,p,q,R,P[1],n[1],Q[1],0,l,1e-8,100)
        d2 = nadcp3(a,b,p,q,R,P[2],n[2],Q[2],0,l,1e-8,100)

        return np.array([d0,d1,d2])

    def hessian_eval(self, Q):
        P = self.segment.P
        n = self.segment.n
        R = self.R
        l = self.segment.l

        a = 2*np.dot(P-Q,n)
        b = np.dot(P-Q,P-Q)

        p = self.p
        q = self.q

        #a,b,p,q,R,Pj  ,vj  ,xj ,t0,t1,error,interval_division
        h00 = nad2iicp3(a,b,p,q,R,P[0],n[0],Q[0],0,l,1e-8,100)
        h11 = nad2iicp3(a,b,p,q,R,P[1],n[1],Q[1],0,l,1e-8,100)
        h22 = nad2iicp3(a,b,p,q,R,P[2],n[2],Q[2],0,l,1e-8,100)

        h01 = nad2ijcp3(a,b,p,q,R,P[0],n[0],Q[0],P[1],n[1],Q[1],0,l,1e-8,100)
        h02 = nad2ijcp3(a,b,p,q,R,P[0],n[0],Q[0],P[2],n[2],Q[2],0,l,1e-8,100)
        h12 = nad2ijcp3(a,b,p,q,R,P[1],n[1],Q[1],P[2],n[2],Q[2],0,l,1e-8,100)

        return np.array([
            [h00,h01,h02],
            [h01,h11,h12],
            [h02,h12,h22]
            ])

class SegmentField2(Field):

    def __init__(self, R, segment, normal, a, b, c, gsl_ws_size=100, max_error=1.0e-8):
        self.skel = segment
        self.P = segment.P
        self.T = segment.n
        self.N = normal
        self.l = segment.l
        self.a = a
        self.b = b
        self.c = c
        self.R = R
        self.n = gsl_ws_size
        self.e = max_error

        if not np.isclose(np.dot(self.T,self.N),0.0):
            raise ValueError("Tangent {} and normal {} are not perpendicular".format(self.T,self.N))

    def eval(self, X):
        return compact_field_eval(X,self.P,self.T,self.N,self.l,self.a,self.b,self.c,self.R,self.n,self.e)


class ArcField2(Field):

    def __init__(self, R, arc, a, b, c, gsl_ws_size=100, max_error=1.0e-8):
        self.skel = arc
        self.C = arc.P
        self.u = arc.n1
        self.v = arc.n2
        self.r = arc.r
        self.phi = arc.phi
        self.a = a
        self.b = b
        self.c = c
        self.R = R
        self.n = gsl_ws_size
        self.e = max_error

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

