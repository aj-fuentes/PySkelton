import numpy as np
import numpy.linalg as nal

class Skeleton(object):

    def __init__(self, lower_bound, upper_bound, initial_weight=1.0, final_weight=1.0):
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self.initial_weight, self.final_weight = initial_weight, final_weight

    def get_bounds(self):
        return (self.lower_bound, self.upper_bound)

    def get_point_at(self, t):
        if self.lower_bound <= t <= self.upper_bound:
            return self.compute_point_at(t)
        else:
            raise ValueError("Skeleton parameter out of bounds: value %f should be in [%f,%f]" % (t,self.lower_bound,self.upper_bound))

    def get_tangent_at(self, t):
        if self.lower_bound <= t <= self.upper_bound:
            return self.compute_tangent_at(t)
        else:
            raise ValueError("Skeleton parameter out of bounds: value %f should be in [%f,%f]" % (t,self.lower_bound,self.upper_bound))

    def compute_point_at(self, t):
        raise NotImplementedError()

    def compute_tangent_at(self, t):
        raise NotImplementedError()

    def get_extremal_points(self):
        return self.get_point_at(self.lower_bound),self.get_point_at(self.upper_bound)

    def get_arc_lenght(self):
        raise NotImplementedError()

class Segment(Skeleton):
    """docstring for Segment"""
    def __init__(self, P, n, l, w0=1.0, w1=1.0):
        super(Segment, self).__init__(0,l,w0,w1)
        self.P = P
        self.n = n
        self.l = l

        if abs(nal.norm(n)-1.0)>1e-10:
            raise ValueError('The norm of the vector n=%s is not 1' % n)

    def compute_point_at(self, t):
        return self.P+t*self.n

    def compute_tangent_at(self,t):
        return self.n

    def get_arc_lenght(self):
        return self.l

    @classmethod
    def make_segment(klass, A, B, w0=1.0, w1=1.0):
        n = B-A
        l = nal.norm(B-A)
        n /= l
        return Segment(A,n,l,w0,w1)

class Arc(Skeleton):

    def __init__(self,P,r,n1,n2,phi,w0=1.0,w1=1.0):
        super(Arc, self).__init__(0,phi,w0,w1)

        cos_phi = np.cos(phi)
        self.u_max = np.sqrt((1-cos_phi)/(1+cos_phi))

        self.P = P
        self.r = r
        self.n1 = n1
        self.n2 = n2
        self.phi = phi

        if abs(nal.norm(n1)-1.0)>1e-10:
            raise ValueError('The norm of the vector n=%s is not 1' % n1)
        if abs(nal.norm(n2)-1.0)>1e-10:
            raise ValueError('The norm of the vector n=%s is not 1' % n2)

    def compute_point_at(self, theta):
        return self.P + self.r*np.cos(theta)*self.n1 + self.r*np.sin(theta)*self.n2

    def compute_tangent_at(self, theta):
        return -np.sin(theta)*self.n1 + np.cos(theta)*self.n2

    def get_arc_lenght(self):
        return self.phi*self.r/(2*np.pi)

class SkeletonGraph(object):
    """docstring for SkeletonGraph"""

    class SkeletonNode(object):

        """docstring for SkeletonNode"""
        def __init__(self, P, idx):
            super(SkeletonGraph.SkeletonNode, self).__init__()
            self.P = P
            self.neighbors = []
            self.idx = idx

        def add_connection(self, node, skel, field):
            P = self.P
            A,B = skel.get_extremal_points()
            if not nal.norm(P-A)<1e-10 and not nal.norm(P-B)<1e-10:
                raise ValueError("The node for %s cannot be connected to the node of %s using the skeleton %s",(P,node.P,skel))
            self.neighbors.append( (node, skel, field) )


    def __init__(self):
        super(SkeletonGraph, self).__init__()

        self.nodes = []


    def add_data(self, skel, field):
        A,B = skel.get_extremal_points()
        nA = None
        nB = None
        for node in self.nodes:
            if nal.norm(A-node.P)<1e-10:
                nA = node
            if nal.norm(B-node.P)<1e-10:
                nB = node
            if nA and nB:
                break
        if nA is None:
            nA = SkeletonGraph.SkeletonNode(A,len(self.nodes))
            self.nodes.append(nA)
        if nB is None:
            nB = SkeletonGraph.SkeletonNode(B,len(self.nodes))
            self.nodes.append(nB)
        nA.add_connection(nB,skel,field)
        nB.add_connection(nA,skel,field)

class SkeletonG1PolyCurve(object):

    def __init__(self):
        super(SkeletonG1PolyCurve, self).__init__()
        self.pieces = []
        self.fields = []

    def add_piece(self, skel, field):

        if self.pieces:
            last = self.pieces[-1]

            A = skel.compute_point_at(skel.lower_bound)
            B = last.compute_point_at(last.upper_bound)
            if nal.norm(A-B)>1e-10:
                raise ValueError("The new piece will break continuity for the skeleton: %s != %s" % (A,B))

            t1 = last.compute_tangent_at(last.upper_bound)
            t2 = skel.compute_tangent_at(last.lower_bound)
            if nal.norm(t1-t2)>1e-10:
                raise ValueError("The new piece will break G1 continuity for the skeleton")

        self.pieces.append(skel)
        self.fields.append(field)




