import numpy as np
import numpy.linalg as nla
from _math import *

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


        vis.add_mesh(ps,fs,name=self.surface_name,color=self.surface_color)

        for i in range(N):
            vis.add_polyline(ps[i*M:i*M+M],name=self.mesh_lines_name,color=self.mesh_lines_color)

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

        # ps = []
        # for m,n in zip(cell1,cell2):
        #     data = [((1.0-t)*m + t*n,P + t*l*v) for t in ts]
        #     qs = [self.field.shoot_ray(Q,u,self.level_set,double_distance=8) for u,Q in data]
        #     ps.extend(qs)
        # ps = np.array(ps)

        data = []
        for m,n in zip(cell1,cell2):
            data.extend( ((1.0-t)*m + t*n,P + t*l*v) for t in ts )
        ps = self.shoot_ray_parallel(data) if self.parallel_ray_shooting else self.shoot_ray(data)

        return ps

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
        idxs = self._match_arc_cells([(p*Fi).A1 for p in cell1],[(p*Fe).A1 for p in cell2])
        cell1 = [cell1[idx] for idx in idxs]

        # ps = []
        # for m,n in zip(cell1,cell2):
        #     m_ = m*Fi
        #     n_ = n*Fe
        #     data = [( (((1.0-t)*m_ + t*n_) * F).A1, C+r*np.cos(t*phi)*u+r*np.sin(t*phi)*v) for t,F in zip(ts,FsT)]
        #     qs = [self.field.shoot_ray(Q,m,self.level_set,double_distance=self.shoot_double_distance) for m,Q in data]
        #     ps.extend(qs)
        # ps = np.array(ps)

        data = []
        for m,n in zip(cell1,cell2):
            m_ = m*Fi
            n_ = n*Fe
            data.extend( ( (((1.0-t)*m_ + t*n_) * F).A1, C+r*np.cos(t*phi)*u+r*np.sin(t*phi)*v) for t,F in zip(ts,FsT) )
        ps = self.shoot_ray_parallel(data)

        return ps

    def _match_arc_cells(self,cell1,cell2):

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


