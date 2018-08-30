import numpy as np
from _math import *

_default_radii = np.ones(2,dtype=float)

class Skeleton(object):

    def __init__(self, l, a, b, c):

        self.l = float(l)
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)

        self._extremities = None

    def get_point_at(self, t):
        raise NotImplementedError()

    def get_tangent_at(self, t):
        raise NotImplementedError()

    def get_normal_at(self, t):
        raise NotImplementedError()

    def get_binormal_at(self, t):
        raise NotImplementedError()

    def get_frame_at(self,t):
        raise NotImplementedError()

    @property
    def extremities(self):
        if self._extremities is None:
            self._extremities = (self.get_point_at(0),self.get_point_at(self.l))
        return self._extremities


class Segment(Skeleton):
    """docstring for Segment"""
    def __init__(self, P, v, l, n, a=_default_radii, b=_default_radii, c=_default_radii):
        super(Segment, self).__init__(l,a,b,c)

        check_unit(v)

        self.P = np.array(P)
        self.v = np.array(v)
        self.l = float(l)
        self.n = np.array(n)

        self.binormal = np.cross(v,n)

        self.F = np.matrix([v,n,self.binormal]).T

    def get_point_at(self, t):
        return self.P+t*self.v

    def get_tangent_at(self,t):
        return self.v

    def get_normal_at(self,t):
        return self.n

    def get_binormal_at(self,t):
        return self.binormal

    def get_frame_at(self,t):
        return self.F

    @staticmethod
    def make_segment(A,B,n,a=_default_radii,b=_default_radii,c=_default_radii):
        v = B-A
        l = norm(v)
        v /= l
        return Segment(A,v,l,n,a,b,c)

class Arc(Skeleton):

    def __init__(self,C,u,v,r,phi,a=_default_radii,b=_default_radii,c=_default_radii):
        super(Arc, self).__init__(phi*r,a,b,c)

        check_unit(u)
        check_unit(v)
        check_perp(u,v)

        self.C = np.array(C)
        self.u = np.array(u)
        self.v = np.array(v)
        self.r = float(r)
        self.phi = float(phi)

        self.binormal = np.cross(u,v)

    def get_point_at(self,t):
        return self.C + self.r*np.cos(t)*self.u + self.r*np.sin(t)*self.v

    def get_tangent_at(self, t):
        return -np.sin(t)*self.u + np.cos(t)*self.v

    def get_normal_at(self, t):
        return -np.cos(t)*self.u - np.sin(t)*self.v

    def get_binormal_at(self,t):
        return self.binormal

    def get_frame_at(self, t):
        return np.matrix([self.get_tangent_at(t),self.get_normal_at(t),self.binormal]).T


