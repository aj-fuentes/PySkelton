import numpy as np
import numpy.linalg as nla
from _math import *

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

        self.surface_name = "surface"
        self.surface_color = "magenta"
        self.mehs_lines_name = "projected_mesh_lines"

    def draw(self,vis):
        for edge in self.scaff.graph.edges:
            if not self.scaff.graph.is_arc_edge(edge):
                self.draw_line_piece(vis,edge)

        for arc in self.scaff.graph.arcs:
            self.draw_arc_piece(vis,arc)

        for i in self.scaff.graph.get_dangling_indices():
            self.draw_dangling_piece(vis,i)

    def draw_line_piece(self,vis,edge):

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

        ps = []
        for m,n in zip(cell1,cell2):
            data = [((1.0-t)*m + t*n,P + t*l*v) for t in np.linspace(0,1.0,self.quads_num+1)]
            qs = [self.field.shoot_ray(Q,u,self.level_set,double_distance=8) for u,Q in data]
            ps.extend(qs)
        ps = np.array(ps)

        N = len(cell1)
        M = self.quads_num+1
        fs =  fs =[[i*M + j, i*M + (j+1),((i+1)%N)*M + (j+1),((i+1)%N)*M + j ] for i in range(N) for j in range(M-1)]

        vis.add_mesh(ps,fs,name=self.surface_name,color=self.surface_color)


    def draw_arc_piece(self,vis,arc):

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
        idxs = self.match_cells([(p*Fi).A1 for p in cell1],[(p*Fe).A1 for p in cell2])
        cell1 = [cell1[idx] for idx in idxs]

        ps = []
        for m,n in zip(cell1,cell2):
            m_ = m*Fi
            n_ = n*Fe
            data = [( (((1.0-t)*m_ + t*n_) * F).A1, C+r*np.cos(t*phi)*u+r*np.sin(t*phi)*v) for t,F in zip(ts,FsT)]
            qs = [self.field.shoot_ray(Q,m,self.level_set,double_distance=8) for m,Q in data]
            ps.extend(qs)
        ps = np.array(ps)

        N = len(cell1)
        M = self.quads_num+1
        fs =  fs =[[i*M + j, i*M + (j+1),((i+1)%N)*M + (j+1),((i+1)%N)*M + j ] for i in range(N) for j in range(M-1)]

        vis.add_mesh(ps,fs,name=self.surface_name,color=self.surface_color)

    def match_cells(self,cell1,cell2):

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

    def draw_dangling_piece(self,vis,i):
        pass
