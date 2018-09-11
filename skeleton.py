import numpy as np
from _math import *
import bisect
import math

class Skeleton(object):

    def __init__(self, l):

        self.l = float(l)
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
        raise np.matrix([self.get_tangent_at(t),self.get_normal_at(t),self.get_binormal_at(t)]).T

    def get_distance(self,X):
        raise NotImplementedError()

    def is_close(self,X):
        return self.get_distance(X)<=self.max_dist

    @property
    def extremities(self):
        if self._extremities is None:
            self._extremities = (self.get_point_at(0),self.get_point_at(self.l))
        return self._extremities


class Segment(Skeleton):
    """docstring for Segment"""
    def __init__(self,P,v,l,n):
        super(Segment, self).__init__(l)

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

    def get_distance(self,X):
        u = X-self.P
        t = np.dot(u,self.v)
        if t<0: t = 0
        elif t>self.l: t = self.l
        return norm(self.get_point_at(t)-X)

    @staticmethod
    def make_segment(A,B,n):
        v = B-A
        l = norm(v)
        v /= l
        return Segment(A,v,l,n)

class Arc(Skeleton):

    def __init__(self,C,u,v,r,phi):
        super(Arc, self).__init__(phi*r)

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
        t /=self.r
        return self.C + self.r*np.cos(t)*self.u + self.r*np.sin(t)*self.v

    def get_tangent_at(self, t):
        return -np.sin(t)*self.u + np.cos(t)*self.v

    def get_normal_at(self, t):
        return -np.cos(t)*self.u - np.sin(t)*self.v

    def get_binormal_at(self,t):
        return self.binormal

    def get_frame_at(self, t):
        return np.matrix([self.get_tangent_at(t),self.get_normal_at(t),self.binormal]).T

    def get_distance(self,X):
        y = np.dot(X-self.C,self.binormal)
        Q = X-y*self.binormal
        q = normalize(Q-self.C)
        t = math.atan2(np.dot(q,self.v),np.dot(q,self.u))
        if t<0: t = 0
        elif t>self.phi: t = self.phi
        return norm(self.get_point_at(t*self.r)-X)

class G1Curve(Skeleton):

    def __init__(self,skels):
        self.ls = [skel.l for skel in skels]
        self.skels = skels
        #compute commulative sum
        for i in range(1,len(skels)): self.ls[i]+=self.ls[i-1]
        super(G1Curve, self).__init__(self.ls[-1])

        for i in range(1,len(skels)):
            T0 = skels[i-1].get_tangent_at(skels[i-1].l)
            T1 = skels[i].get_tangent_at(0)
            assert np.isclose(nla.norm(T0-T1),0.0),"The piece # {0} breaks continuity. Last tangent {1} != new tangent {2}".format(i,T0,T1)

        self._compute_rot_angles()

    def _compute_rot_angles(self):
        self.angles = [0.0] #store the rotation angles
        for i in range(len(1,self.skels)):
            #get the frame of previous piece at the end
            N0 = self.skels[i-1].get_normal_at(self.skels[i-1].l)
            B0 = self.skels[i-1].get_binormal_at(self.skels[i-1].l)

            #get the frame of current piece at the beggining
            N1 = self.skels[i].get_normal_at(0.0)
            B1 = self.skels[i].get_binormal_at(0.0)

            #last angle
            angle = self.angles[-1]

            #previous frame was rotated, thus recompute normal
            N = N0*math.cos(angle) + B0*math.sin(angle)

            #compute the angle between the current frame and the recomputed previous normal
            angle = math.atan2(np.dot(N,B1),np.dot(N,N1))
            self.angles.append(angle)

    def find_piece(self,t):
        return bisect.bisect_left(self.ls,t)

    def get_point_at(self,t):
        i = self.find_piece(t)
        skel,t = self.skels[i],t-self.ls[i]
        return skel.get_point_at(t)

    def get_tangent_at(self, t):
        i = self.find_piece(t)
        skel,t = self.skels[i],t-self.ls[i]
        return skel.get_tangent_at(t)

    def get_normal_at(self, t):
        i = self.find_piece(t)
        skel,t = self.skels[i],t-self.ls[i]
        angle = self.angles[i]
        return skel.get_normal_at(t)*math.cos(angle) + skel.get_binormal_at(t)*math.sin(angle)

    def get_binormal_at(self,t):
        i = self.find_piece(t)
        skel,t = self.skels[i],t-self.ls[i]
        angle = self.angles[i]
        return -skel.get_normal_at(t)*math.sin(angle) + skel.get_binormal_at(t)*math.cos(angle)

    def get_distance(self,X):
        return min(skel.get_distance(X) for skel in skels)
