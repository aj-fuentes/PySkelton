import sys
import itertools
import time
import multiprocessing as mp

import numpy as np
import numpy.linalg as nla

from ._math import *
from .visualization import skel_palette
from .graph import Graph
from .field import SegmentField,ArcField,G1Field,MultiField
from .skeleton import Segment,Arc

def graph_and_pieces_from_field(field):
    pieces = []
    g = Graph()

    f = field
    if isinstance(f,SegmentField):
        i,j = g.add_node(f.skel.extremities[0]),g.add_node(f.skel.extremities[1])
        g.add_edge(i,j)

        pieces.append((f.skel,[i,j]))

    elif isinstance(f,ArcField):
        nodes = arc_to_nodes(f.skel.C,f.skel.u,f.skel.v,f.skel.r,f.skel.phi)
        idxs = [g.add_node(node) for node in nodes]
        g.add_edge(idxs[0],idxs[1])
        g.add_edge(idxs[1],idxs[2])
        g.add_edge(idxs[2],idxs[3])

        pieces.append((f.skel,idxs))

    elif isinstance(f,G1Field):
        full_idxs = []
        for skel in f.skel.skels:
            if isinstance(skel,Segment):
                i,j = g.add_node(skel.extremities[0]),g.add_node(skel.extremities[1])
                g.add_edge(i,j)
                full_idxs.append(i)
                full_idxs.append(j)
            elif isinstance(skel,Arc):
                nodes = arc_to_nodes(skel.C,skel.u,skel.v,skel.r,skel.phi)
                idxs = [g.add_node(node) for node in nodes]
                g.add_edge(idxs[0],idxs[1])
                g.add_edge(idxs[1],idxs[2])
                g.add_edge(idxs[2],idxs[3])
                full_idxs.extend(idxs)
            else: assert False,"G1 curve pieces must be arcs or segments"

        pieces.append((f.skel,[full_idxs[0],full_idxs[1],full_idxs[-2],full_idxs[-1]]))

    elif isinstance(f,MultiField):
        for f in f.fields:
            g0,pieces0 = graph_and_pieces_from_field(f)
            for i0,j0 in g0.edges:
                i,j = g.add_node(g0.nodes[i0]),g.add_node(g0.nodes[j0])
                g.add_edge(i,j)
            for skel0,idxs0 in pieces0:
                idxs = [
                    g.find_node_from_point(g0.nodes[idxs0[0]]),
                    g.find_node_from_point(g0.nodes[idxs0[1]]),
                    g.find_node_from_point(g0.nodes[idxs0[-2]]),
                    g.find_node_from_point(g0.nodes[idxs0[-1]])
                ]
                pieces.append((skel0,idxs))

    return g,pieces

_extra_dist = np.array([5.0,0.0,0.0])

__last_r = None
def _shoot_ray(args):
    global __last_r
    f,(u,Q,guess_r),level_set,double_distance = args
    if __last_r: guess_r = 1.1*__last_r
    point = f.shoot_ray(Q,u,level_set,double_distance,guess_r)
    grad  = f.gradient_eval(point)
    __last_r = nla.norm(point-Q)
    return point,grad

class Mesher(object):
    """docstring for Mesher"""
    def __init__(self, scaff, field, pieces):
        super(Mesher, self).__init__()
        self.scaff = scaff
        self.field = field
        self.pieces = pieces

        self.quads_num = 8
        self.cap_quads_num = 8
        self.max_quads_size = 0.0 #defines the quads_num
        self.split_output = False
        self.level_set = 0.1
        self.shoot_double_distance = 8

        self.count_field_evals = False #whether to count field evaluations
        self.field_evals_counter = None #counter for field evaluations
        self.multi_field_evals_counter = None #counter for multi_field evaluations
        self.field_evals_time = None #save total time for field evaluations
        self.ray_shooting_counter = None #counter for ray shootings
        self.ray_shooting_timer = 0.0 #save real time (parallelims) of ray shooting

        self.workers_pool = None #pool of workers for parallel ray shooting

        self.surface_name = "surface"
        self.surface_color = "magenta"
        self.mesh_lines_name = "projected_mesh_lines"
        self.mesh_lines_color = "cyan"

        self.parallel_ray_shooting = True
        self.parallel_piece_computing = False

        self.piece_ps = []
        self.dangling_ps = []

        self.show_gradients = False
        self.show_normals = False
        self.compute_normals = False

    def init_pool_workers(self):
        self.ray_shooting_counter = 0
        self.ray_shooting_timer = 0

        self.field_evals_counter = None
        self.multi_field_evals_counter = None
        self.field_evals_time = None
        if self.count_field_evals:
            self.field_evals_counter = mp.Value('i',0)
            self.multi_field_evals_counter = mp.Value('i',0)
            self.field_evals_time = mp.Value('d',0)

        def _init_counter(pk_module, eval_counter, multi_eval_counter, eval_time):
            pk_module.field._eval_counter = eval_counter
            pk_module.field._multi_eval_counter = multi_eval_counter
            pk_module.field._eval_time = eval_time

        _module_name,_,_ = __name__.partition(".")

        self.workers_pool = mp.Pool(initializer=_init_counter,initargs=(sys.modules[_module_name],self.field_evals_counter,self.multi_field_evals_counter,self.field_evals_time))

    def compute(self,print_summary=False):
        self.init_pool_workers()

        if not self.piece_ps:
            self.piece_ps = [self.compute_piece(skel,nodes) for skel,nodes in self.pieces]

        if not self.dangling_ps:
            self.dangling_ps = [self.compute_dangling_piece(i) for i in self.scaff.graph.get_dangling_indices()]

        self.workers_pool.close()
        self.workers_pool.join()

        if print_summary:
            self.print_summary_data()

    def draw(self,vis):

        self.compute(print_summary=False)

        n_colors = len(skel_palette)
        for i,(ps,(skel,_)) in enumerate(zip(self.piece_ps,self.pieces)):
            self.draw_piece(vis,ps,skel)
            vis.add_polyline([skel.get_point_at(t) for t in np.linspace(0.0,skel.l,int(skel.l/0.2)+2)],name="skel_splines_mesher".format(i),color=skel_palette[i%n_colors])

        for ps in self.dangling_ps:
            self.draw_piece_cap(vis,ps)

        return vis

    def shoot_ray_parallel(self,data):

        self.ray_shooting_counter += len(data)

        fs = itertools.repeat(self.field)
        ls = itertools.repeat(self.level_set)
        ds = itertools.repeat(self.shoot_double_distance)

        args = zip(fs,data,ls,ds)

        return self.workers_pool.map(_shoot_ray,args)

    def shoot_ray(self,data):
        return [self.field.shoot_ray(Q,v,self.level_set,self.shoot_double_distance,guess_r) for (v,Q,guess_r) in data]

    def draw_piece(self,vis,pgs,skel,color=None,name=None,draw_mesh_lines=True):
        ps = [pg[0] for pg in pgs]
        grads = [pg[1] for pg in pgs]
        M = self.get_shooting_num(skel)
        N = len(ps)//M
        fs = [[i*M + j, i*M + (j+1),((i+1)%N)*M + (j+1),((i+1)%N)*M + j ] for i in range(N) for j in range(M-1)]

        suffix = str(id(ps)) if self.split_output else ""
        if color is None:
            color = self.surface_color
        if name is None:
            name = self.surface_name+suffix

        #check for the right order in the faces
        new_fs = []
        for f in fs:
            u0,u1 = ps[f[1]]-ps[f[0]],ps[f[3]]-ps[f[0]]
            u0,u1 = normalize(u0),normalize(u1)
            if np.dot(grads[f[0]],np.cross(u0,u1))>0.0:
                #reverse the faces!!
                f = [f[0],f[3],f[2],f[1]]
            new_fs.append(f)
        fs = new_fs
        vis.add_mesh(ps,fs,name=name+"_mesher",color=color)

        #add normals to the visualization
        vis.add_normals([-normalize(grad) for grad in grads],name=name+"_mesher")

        if draw_mesh_lines:
            for i in range(N):
                vis.add_polyline(ps[i*M:i*M+M],name=self.mesh_lines_name+suffix+"_mesher",color=self.mesh_lines_color)

        if self.show_gradients or self.show_normals:
            for X,grad in zip(ps,grads):
                if self.show_gradients:
                    vis.add_polyline([X,X+grad],name="gradients_mesher",color="yellow")
                else:
                    vis.add_polyline([X,X-0.1*normalize(grad)],name="normals_mesher",color="blue")

        return vis

    def draw_piece_cap(self,vis,pgs,color=None,name=None,draw_mesh_lines=True):
        ps = [pg[0] for pg in pgs]
        grads = [pg[1] for pg in pgs]

        M = self.cap_quads_num + 1
        N = len(ps)//M
        fs = [[i*M + j, i*M + (j+1),((i+1)%N)*M + (j+1),((i+1)%N)*M + j ] for i in range(N) for j in range(M-1)]

        suffix = str(id(ps)) if self.split_output else ""
        if color is None:
            color = self.surface_color
        if name is None:
            name = self.surface_name+suffix
        vis.add_mesh(ps,fs,name=name+"_mesher",color=color)

        #add normals to the visualization
        vis.add_normals([-normalize(grad) for grad in grads],name=name+"_mesher")

        if draw_mesh_lines:
            for i in range(N):
                vis.add_polyline(ps[i*M:i*M+M],name=self.mesh_lines_name+suffix+"_mesher",color=self.mesh_lines_color)

        if self.show_gradients or self.show_normals:
            for X,grad in zip(ps,grads):
                if self.show_gradients:
                    vis.add_polyline([X,X+grad],name="gradients_mesher",color="yellow")
                else:
                    vis.add_polyline([X,X-0.1*normalize(grad)],name="normals_mesher",color="blue")

        return vis

    def draw_isolated_piece(self,vis,f,vecs_num):
        ts = np.linspace(0.0,f.skel.l,self.get_shooting_num(f.skel))
        NBs = [[f.skel.get_normal_at(t),f.skel.get_binormal_at(t)] for t in ts]
        Ps = [f.skel.get_point_at(t) for t in ts]

        vs = [np.array([1.0,0.0])*np.cos(phi) + np.array([0.0,1.0])*sin(phi) for phi in np.linspace(0.0,2.0*np.pi,vecs_num,endpoint=False)]
        vs /= nla.norm(vs,axis=1)[:,None]

        ps = [
            f.shoot_ray(P,v[0]*N+v[1]*B,self.level_set,self.shoot_double_distance) for v in vs for P,(N,B) in zip(Ps,NBs)
        ]
        pgs = [(p,f.gradient_eval(p)) for p in ps]
        self.draw_piece(vis,pgs,skel=f.skel,color="blue",name="isolated_pieces",draw_mesh_lines=False)

    def compute_dangling_piece(self,i):
        P = self.scaff.graph.nodes[i]
        e = self.scaff.graph.incident_edges[i][0]

        cell = self.scaff.node_cells[i][e][:-1]

        j = e[0]
        if j==i: j = e[1]
        v = normalize(P-self.scaff.graph.nodes[j])

        N = self.cap_quads_num + 1
        ts = np.linspace(0.0,np.pi/2.0,N)

        #check for the right order in the faces
        u0,u1 = cell[0],cell[1]
        if np.dot(v,np.cross(u0,u1))>0.0:
            ts = list(reversed(ts))


        Ps = itertools.repeat(P) #Points
        _r = None
        rs = itertools.repeat(_r)
        data = []
        for u in cell:
            assert np.isclose(np.dot(u,v),0.0),"Dangling cell vector not perp to skel"
            vecs = [normalize(u*np.cos(t) + v*np.sin(t)) for t in ts]
            data.extend(list(zip(vecs,Ps,rs)))

        if self.parallel_ray_shooting:
            start = time.time()
            res = self.shoot_ray_parallel(data)
            end = time.time()
            self.ray_shooting_timer += end-start
            return res
        else:
            return self.shoot_ray(data)

    def get_shooting_num(self,skel):
        N = self.quads_num+1
        if self.max_quads_size>0.0:
            N = max(N,int(skel.l/self.max_quads_size))
        print("N {}, skel length {}".format(N,skel.l))
        return N

    def compute_piece(self,skel,nodes):

        #start and ending edges for the piece
        e1 = make_edge(nodes[0],nodes[1])
        e2 = make_edge(nodes[-2],nodes[-1])

        #get cells
        cell1 = self.scaff.node_cells[nodes[0]][e1][:-1]
        cell2 = self.scaff.node_cells[nodes[-1]][e2][:-1]

        if not np.isclose(nla.norm(self.scaff.graph.nodes[nodes[0]]-skel.extremities[0]),0.0):
            cell1,cell2 = cell2,cell1

        #compute extra data associated to the piece
        N = self.get_shooting_num(skel)
        ts = np.linspace(0.0,1.0,N)

        Ps = [skel.get_point_at(t*skel.l) for t in ts] #Points
        # _r = max(max(skel.field.b),max(skel.field.c))
        _r = None
        rs = [_r for t in ts]
        FsT = [skel.get_frame_at(t*skel.l).transpose() for t in ts] #Frames transposed
        Fi,Fe = skel.get_frame_at(0.0),skel.get_frame_at(skel.l) #initial and ending Frames

        #convert cells to local coordinates
        local_cell1 = [(p*Fi).A1 for p in cell1]
        local_cell2 = [(p*Fe).A1 for p in cell2]

        #match cells (the second cell must have inverted tangent directions!)
        idxs = self._match_cells(local_cell1,local_cell2)

        #reorder cell1 to match the links
        local_cell1 = [local_cell1[idx] for idx in idxs]

        #compute the shooting vectors data
        data = []
        for m,n,m2,n2 in zip(local_cell1,local_cell2,cell1,cell2):
            vecs = [normalize((1.0-t)*m + t*n) for t in ts]
            vecs = [(v * Ft).A1 for Ft,v in zip(FsT,vecs)]
            data.extend(list(zip(vecs,Ps,rs)))

        if self.parallel_ray_shooting:
            start = time.time()
            res = self.shoot_ray_parallel(data)
            end = time.time()
            self.ray_shooting_timer += end-start
            return res
        else:
            return self.shoot_ray(data)

    def _match_cells(self,cell1,cell2):

        min_d = np.inf
        min_i = -1

        nn = len(cell1)
        idxs = list(range(nn))
        reverse = False

        cell2 = [p+_extra_dist for p in cell2]
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
        res = [((-j if reverse else j)+min_i)%nn for j in idxs]
        return res

    def print_summary_data(self,out=sys.stdout):
        mfev = self.multi_field_evals_counter.value
        fev = self.field_evals_counter.value
        rsh = self.ray_shooting_counter
        avg = mfev/rsh
        print(f"ray-shootings: {rsh}\nfield evals: {mfev}\naverage evals per ray-shooting: {avg}\nintegral evals: {fev}",file=out)

        ttime = self.field_evals_time.value
        tavg = ttime/fev
        print(f"total integral time: {ttime}s\naverage time per intergal: {tavg*1000:.2}ms",file=out)

        rstime = self.ray_shooting_timer
        print(f"ray-shooting real time:{rstime}s",file=out)
