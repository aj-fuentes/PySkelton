import numpy as np
import numpy.linalg as nla
from _math import *
import math

from multiprocessing import Pool
import itertools

def _shoot_ray(args):
    f,(u,Q),level_set,double_distance = args
    return f.shoot_ray(Q,u,level_set,double_distance)

def _compute_line_piece(args):
    self,edge = args
    return self.compute_line_piece(edge)

def _compute_arc_piece(args):
    self,arc = args
    return self.compute_arc_piece(edge)

def _compute_dangling_piece(args):
    self,i = args
    return self.compute_dangling_piece(edge)

class Mesher(object):
    """docstring for Mesher"""
    def __init__(self, scaff, field):
        super(Mesher, self).__init__()
        self.scaff = scaff
        self.field = field

        self.quads_num = 8
        self.cap_quads_num = 8
        self.split_output = False
        self.level_set = 0.1
        self.shoot_double_distance = 8

        self.workers_pool = Pool(4)
        self.workers_pool2 = Pool(4)

        self.surface_name = "surface"
        self.surface_color = "magenta"
        self.mesh_lines_name = "projected_mesh_lines"
        self.mesh_lines_color = "cyan"

        self.parallel_ray_shooting = True
        self.parallel_piece_computing = False


        self.line_ps = []
        self.arc_ps = []
        self.dangling_ps = []


    def compute(self):
        es = [edge for edge in self.scaff.graph.edges if not self.scaff.graph.is_arc_edge(edge)]
        arcs = self.scaff.graph.arcs

        if not self.line_ps:
            if self.parallel_piece_computing:
                args = itertools.izip(itertools.repeat(self),es)
                self.line_ps = self.workers_pool2.map(_compute_line_piece,args)
            else:
                self.line_ps = [self.compute_line_piece(edge) for edge in es]

        if not self.arc_ps:
            if self.parallel_piece_computing:
                args = itertools.izip(itertools.repeat(self),arcs)
                self.arc_ps = self.workers_pool2.map(_compute_arc_piece,args)
            else:
                self.arc_ps = [self.compute_arc_piece(arc) for arc in arcs ]

        # if not self.dangling_ps:
            # self.dangling_ps = [self.compute_dangling_piece(i) for i in self.scaff.graph.get_dangling_indices()]

    def draw(self,vis):

        self.compute()

        for ps in self.line_ps:
            self.draw_piece(vis,ps)
        for ps in self.arc_ps:
            self.draw_piece(vis,ps)
        # for ps in self.dangling_ps:
        #     self.draw_piece(vis,ps)

        # for skel,angle in zip(self.field.skel.skels,self.field.skel.angles):
        #     vis.add_polyline([skel.extremities[0],skel.extremities[0]+skel.get_normal_at(0.0)],name="normals",color="blue")
        #     vis.add_polyline([skel.extremities[1],skel.extremities[1]+skel.get_normal_at(skel.l)],name="normals",color="blue")

        #normals
        # for skel,angle in zip(self.field.skel.skels,self.field.skel.angles):
        #     P = skel.extremities[0]

        #     n = skel.get_normal_at(0.0)
        #     b = skel.get_binormal_at(0.0)
        #     assert np.isclose(np.dot(b,n),0.0),"Unrotated initial Normal and Binormal are not perpendicular"

        #     ni =  n*math.cos(angle) + b*math.sin(angle)
        #     bi = -n*math.sin(angle) + b*math.cos(angle)
        #     assert np.isclose(np.dot(bi,ni),0.0),"Initial Normal and Binormal are not perpendicular"

        #     vis.add_polyline([P,P+ni],name="normals",color="blue")
        #     vis.add_polyline([P,P+bi],name="binormals",color="green")

        #     P = skel.extremities[1]

        #     n = skel.get_normal_at(skel.l)
        #     b = skel.get_binormal_at(skel.l)
        #     assert np.isclose(np.dot(b,n),0.0),"Unrotated final Normal and Binormal are not perpendicular"

        #     ne =  n*math.cos(angle) + b*math.sin(angle)
        #     be = -n*math.sin(angle) + b*math.cos(angle)
        #     assert np.isclose(np.dot(be,ne),0.0),"Final Normal and Binormal are not perpendicular"
        #     vis.add_polyline([P,P+ne],name="normals",color="blue")
        #     vis.add_polyline([P,P+be],name="binormals",color="green")


        return vis


    def shoot_ray_parallel(self,data):

        fs = itertools.repeat(self.field)
        ls = itertools.repeat(self.level_set)
        ds = itertools.repeat(self.shoot_double_distance)

        args = itertools.izip(fs,data,ls,ds)

        return self.workers_pool.map(_shoot_ray,args)

    def shoot_ray(self,data):
        return [self.field.shoot_ray(Q,v,self.level_set,self.shoot_double_distance) for (v,Q) in data]

    def draw_piece(self,vis,ps):
        M = self.quads_num+1
        N = len(ps)/M
        fs =  fs =[[i*M + j, i*M + (j+1),((i+1)%N)*M + (j+1),((i+1)%N)*M + j ] for i in range(N) for j in range(M-1)]

        suffix = str(id(ps)) if self.split_output else ""

        vis.add_mesh(ps,fs,name=self.surface_name+suffix,color=self.surface_color)

        for i in range(N):
            vis.add_polyline(ps[i*M:i*M+M],name=self.mesh_lines_name+suffix,color=self.mesh_lines_color)

        return vis

    def compute_dangling_piece(self,i):
        pass

    def compute_line_piece(self,edge):

        #data for line segment
        P = self.scaff.graph.nodes[edge[0]]
        v = self.scaff.graph.nodes[edge[1]]-P
        l = nla.norm(v)
        v /= l

        #get cells
        cell1 = self.scaff.node_cells[edge[0]][edge][:-1]
        cell2 = self.scaff.node_cells[edge[1]][edge][:-1]

        #reorder cell1 to match the links
        cell1 = [cell1[idx] for idx in self.scaff.links[edge]]

        ts = np.linspace(0,1.0,self.quads_num+1)

        #compute the shooting vectors data
        data = []
        for m,n in zip(cell1,cell2):
            data.extend( ((1.0-t)*m + t*n,P + t*l*v) for t in ts )

        if self.parallel_ray_shooting:
            return self.shoot_ray_parallel(data)
        else:
            return self.shoot_ray(data)

    def compute_arc_piece(self,arc):

        #data for arc
        C,u,v,r,phi = nodes_to_arc(*[self.scaff.graph.nodes[i] for i in arc])
        arc_normal = np.cross(u,v)

        #start and ending edges for the arc
        e1 = make_edge(arc[0],arc[1])
        e2 = make_edge(arc[2],arc[3])

        #get cells
        cell1 = self.scaff.node_cells[arc[0]][e1][:-1]
        cell2 = self.scaff.node_cells[arc[3]][e2][:-1]

        #compute extra data associated to the arc
        ts = np.linspace(0,1.0,self.quads_num+1)
        Fs = [np.matrix([-np.sin(th)*u+np.cos(th)*v,arc_normal,np.cos(th)*u+np.sin(th)*v]).transpose() for th in phi*ts] #Frames
        FsT = [F.transpose() for F in Fs]
        Fi,Fe = Fs[0],Fs[-1]

        #reorder cell1 to match the links
        idxs = self._match_cells([(p*Fi).A1 for p in cell1],[(p*Fe).A1 for p in cell2])
        cell1 = [cell1[idx] for idx in idxs]

        #compute the shooting vectors data
        data = []
        for m,n in zip(cell1,cell2):
            m_ = m*Fi
            n_ = n*Fe
            data.extend( ( (((1.0-t)*m_ + t*n_) * F).A1, C+r*np.cos(t*phi)*u+r*np.sin(t*phi)*v) for t,F in zip(ts,FsT) )

        if self.parallel_ray_shooting:
            return self.shoot_ray_parallel(data)
        else:
            return self.shoot_ray(data)

    def _match_cells(self,cell1,cell2):

        min_d = np.inf
        min_i = -1

        nn = len(cell1)
        idxs = range(nn)
        reverse = False

        for i in idxs:
            d = sum((norm(cell2[j]-cell1[(j+i)%nn]) for j in idxs))
            if d<min_d:
                min_d = d
                min_i = i
                reverse = False

            d = sum((norm(cell2[j]-cell1[(-j+i)%nn]) for j in idxs))
            if d<min_d:
                min_d = d
                min_i = i
                reverse = True

        return [((-j if reverse else j)+min_i)%nn for j in idxs]
