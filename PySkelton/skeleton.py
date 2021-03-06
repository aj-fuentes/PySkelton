import numpy as np
from ._math import *
import bisect
import math
from . import biarcs as ba

class Skeleton(object):

    def __init__(self, l):

        self.l = float(l)
        self._extremities = None
        self.field = None #to recover the field that directly uses this skeleton

    def get_point_at(self, t):
        raise NotImplementedError()

    def get_tangent_at(self, t):
        raise NotImplementedError()

    def get_normal_at(self, t):
        raise NotImplementedError()

    def get_binormal_at(self, t):
        raise NotImplementedError()

    def get_frame_at(self,t):
        return np.matrix([self.get_tangent_at(t),self.get_normal_at(t),self.get_binormal_at(t)]).T

    def get_distance(self,X):
        raise NotImplementedError()

    def is_close(self,X):
        return self.get_distance(X)<=self.max_dist

    @property
    def extremities(self):
        if self._extremities is None:
            self._extremities = (self.get_point_at(0.0),self.get_point_at(self.l))
        return self._extremities

class Segment(Skeleton):
    """docstring for Segment"""
    def __init__(self,P,v,l,n):
        super(Segment, self).__init__(l)

        check_unit(v)
        check_unit(n)
        assert np.isclose(np.dot(v,n),0.0),"Non perp tangent - normal"

        self.P = np.array(P)
        self.v = np.array(v)
        self.l = float(l)
        self.n = np.array(n)

        self.binormal = normalize(np.cross(v,n))
        check_unit(self.binormal)

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
        A,B,n = np.array(A),np.array(B),np.array(n)
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
        return self.C + self.r*math.cos(t)*self.u + self.r*math.sin(t)*self.v

    def get_tangent_at(self, t):
        t /=self.r
        return -math.sin(t)*self.u + math.cos(t)*self.v

    def get_normal_at(self, t):
        t /=self.r
        return -math.cos(t)*self.u - math.sin(t)*self.v

    def get_binormal_at(self,t):
        return self.binormal

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
        assert sum(skel.l for skel in skels)==self.ls[-1],"Wrong g1curve length"
        super(G1Curve, self).__init__(self.ls[-1])

        for i in range(1,len(skels)):
            T0 = skels[i-1].get_tangent_at(skels[i-1].l)
            T1 = skels[i].get_tangent_at(0.0)
            assert np.isclose(nla.norm(T0-T1),0.0),"The piece # {0} breaks continuity. Last tangent {1} != new tangent {2} (diff norm={3} )".format(i,T0,T1,nla.norm(T0-T1))

        self._compute_rot_angles()

    def _compute_rot_angles(self):
        self.angles = [0.0] #store the rotation angles
        for i in range(1,len(self.skels)):
            #get the frame of previous piece at the end
            T0 = self.skels[i-1].get_tangent_at(self.skels[i-1].l)
            N0 = self.skels[i-1].get_normal_at(self.skels[i-1].l)
            B0 = self.skels[i-1].get_binormal_at(self.skels[i-1].l)

            assert np.isclose(nla.norm(T0),1.0),"Non unit tangent"
            assert np.isclose(nla.norm(N0),1.0),"Non unit normal"
            assert np.isclose(nla.norm(B0),1.0),"Non unit binormal"

            assert np.isclose(np.dot(T0,N0),0.0),"Non perp tangent - normal"
            assert np.isclose(np.dot(T0,B0),0.0),"Non perp tangent - binormal"
            assert np.isclose(np.dot(N0,B0),0.0),"Non perp normal - binormal"


            #get the frame of current piece at the beginning
            T1 = self.skels[i].get_tangent_at(0.0)
            N1 = self.skels[i].get_normal_at(0.0)
            B1 = self.skels[i].get_binormal_at(0.0)

            assert np.isclose(nla.norm(T1),1.0),"Non unit tangent"
            assert np.isclose(nla.norm(N1),1.0),"Non unit normal"
            assert np.isclose(nla.norm(B1),1.0),"Non unit binormal"

            assert np.isclose(np.dot(T1,N1),0.0),"Non perp tangent - normal"
            assert np.isclose(np.dot(T1,B1),0.0),"Non perp tangent - binormal"
            assert np.isclose(np.dot(N1,B1),0.0),"Non perp normal - binormal"

            assert np.isclose(nla.norm(T0-T1),0.0),"Tangents do not coincide"

            #last angle
            angle = self.angles[-1]

            #previous frame was rotated, thus recompute normal
            N = N0*math.cos(angle) + B0*math.sin(angle)
            assert np.isclose(nla.norm(N),1.0),"Non unit rotated normal"

            #compute the angle between the current frame and the recomputed previous normal
            angle = math.atan2(np.dot(N,B1),np.dot(N,N1))
            self.angles.append(angle)

            assert np.allclose(N,math.cos(angle)*N1+math.sin(angle)*B1),"Atan2 computation is wrong"

            Ne = math.cos(self.angles[-2])*N0 + math.sin(self.angles[-2])*B0
            Ni = math.cos(self.angles[-1])*N1 + math.sin(self.angles[-1])*B1

            assert np.isclose(nla.norm(Ne),1.0),"Non unit normal"
            assert np.isclose(nla.norm(Ni),1.0),"Non unit normal"

            assert np.isclose(nla.norm(Ne-Ni),0.0),"Frames not rotated correctly to ensure continuity"

            Be = -math.sin(self.angles[-2])*N0 + math.cos(self.angles[-2])*B0
            assert np.isclose(nla.norm(Be-np.cross(T0,Ne)),0.0),"Incorrect frame at the end"

            Bi = -math.sin(self.angles[-1])*N1 + math.cos(self.angles[-1])*B1
            assert np.isclose(nla.norm(Bi-np.cross(T1,Ni)),0.0),"Incorrect frame at the beginning"

            assert np.isclose(nla.norm(Be-Bi),0.0),"Frames not rotated correctly to ensure continuity"

            assert np.isclose(np.dot(Ne,Be),0.0),"Normal and Binormal are not perpendicular"
            assert np.isclose(np.dot(Ni,Bi),0.0),"Normal and Binormal are not perpendicular"

    def find_piece(self,t):
        return bisect.bisect_left(self.ls,t)

    def get_point_at(self,t):
        i = self.find_piece(t)
        skel = self.skels[i]
        if i>0: t-=self.ls[i-1]
        return skel.get_point_at(t)

    def get_tangent_at(self, t):
        i = self.find_piece(t)
        skel = self.skels[i]
        if i>0: t-=self.ls[i-1]
        return skel.get_tangent_at(t)

    def get_normal_at(self, t):
        i = self.find_piece(t)
        skel = self.skels[i]
        if i>0: t-=self.ls[i-1]
        angle = self.angles[i]
        return skel.get_normal_at(t)*math.cos(angle) + skel.get_binormal_at(t)*math.sin(angle)

    def get_binormal_at(self,t):
        i = self.find_piece(t)
        skel = self.skels[i]
        if i>0: t-=self.ls[i-1]
        angle = self.angles[i]
        return -skel.get_normal_at(t)*math.sin(angle) + skel.get_binormal_at(t)*math.cos(angle)

    def get_distance(self,X):
        return min(skel.get_distance(X) for skel in self.skels)

    @classmethod
    def compute_from_points(klass,As,max_tangent_length=1.0,initial_normal=None):
        aa,_,_2 = ba.circular_spline_from_points(As,max_tangent_length=max_tangent_length)

        skel_pieces = []
        nsegs = 0
        narcs = 0
        for i,(C,u,v,r,phi) in enumerate(aa):
            #if radius is infty then it's a segment
            if r==np.inf:
                #C is P, u is tangent, phi is length, and v is normal of the segment
                if (i==0) and (not (initial_normal is None)):
                    v = initial_normal
                seg = Segment(C,u,phi,v)
                skel_pieces.append(seg)
                nsegs+=1
                # print("segment {} {} {} {}".format(seg.P,seg.v,seg.l,seg.n))
            else:
                arc = Arc(C,u,v,r,phi)
                skel_pieces.append(arc)
                narcs+=2
        print("{} segments, {} arcs".format(nsegs,narcs))

        skel = G1Curve(skel_pieces)
        return skel
