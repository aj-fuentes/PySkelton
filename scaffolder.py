import collections as co
import time
import re
import subprocess as sp
import os

import numpy as np
import numpy.linalg as nla

from graph import Graph
from chull import ConvexHull
import visualization as visual

from _math import *

class Scaffolder(object):
    """Implementation of Scaffolding skeletons using spherical Voronoi diagrams"""

    def __init__(self, graph):
        super(Scaffolder, self).__init__()
        self.graph = graph

        #intersections of the edges with unit sphere
        #same order as nodes
        self.avs = None

        #convex hulls, same order as nodes
        self.chs = None
        self.cos_merge=0.999

        #cells per node
        self.node_cells = [{edge:None for edge in edges} for edges in self.graph.incident_edges]


        #indices for the LP
        self.lp_joint_indices = []
        self.lp_dangling_indices = []
        self.lp_articulation_indices = []

        #output details
        self.split_output = False

        #better colors for the patches
        self.palette = None

        #radii for the nodes
        self.radii = [1.0]*len(graph.nodes)

        #variations
        self.regular = False
        self.symmetric = False
        self.symmetries = []

        #store the number of vertices per valency
        self.vertex_valencies = co.defaultdict(lambda : 0)

        #name for file output
        self.name = "scaffold"

        #subdivisions for the quads
        self.quad_subdiv = 1

        #for less subdivisions
        self.less_subdivs = True
        self.min_subdivs = 4

        #timing
        self.running_time = 0
        self.lp_time = 0
        self.ch_time = 0

        #cross profiles
        self.cross_profiles = set()


    def compute_scaffold(self):

        clock = time.clock()
        self.compute_intersections()

        ch_clock = time.clock()
        self.compute_convex_hulls()
        self.ch_time += (time.clock() - ch_clock)*1000

        lp_clock = time.clock()
        self.generate_IP()
        self.solve_IP()
        self.lp_time += (time.clock() - lp_clock)*1000

        self.create_cells()
        self.create_links()
        self.running_time += (time.clock() - clock)*1000

    def set_node_radius(self,idx,r):
        self.radii[idx] = r

    def set_radii(self,rs):
        if len(rs)!=len(self.graph.nodes):
            raise ValueError("Length of radii list is not equal to the number of nodes")
        self.radii = rs

    def set_all_radius(self,r):
        self.radii = [r]*len(self.graph.nodes)

    def set_regular(self, reg):
        self.regular = reg

    def set_symmetric(self, sym):
        self.symmetric = sym

    def add_symmetry(self, s):
        self.symmetries.append(s)

    def clear_symmetries(self):
        self.symmetries = []

    def set_palette(self,palette):
        self.palette = palette

    def generate_symmetric_equations(self,n,idx_T):
        T = self.symmetries[idx_T]
        ch = self.chs[n]
        for edge in ch.edges:
            i,j = edge

            eq_name = "symmetry_%d_%d_%d_%d" % (idx_T,n,i,j)

            left = "x%di%dj%d" % (n,i,j)
            Tn = T[n]

            #find the edge giving the intersection i in the convex hull
            ei = self.graph.incident_edges[n][i]
            #take symmetric edge
            Tei = T[ei[0]],T[ei[1]]
            if Tei[1]<Tei[0]:
                Tei = Tei[1],Tei[0]

            # print "node %d edge %d %d" % (n,i,j)
            # print "Tn %d" % Tn
            # print self.graph.incident_edges[Tn]

            #find the intersection of the symmetric edge on the symmetric node
            #this position gives the position on the symmetric convex hull
            Ti = self.graph.incident_edges[Tn].index(Tei)

            #do the same for j
            ej = self.graph.incident_edges[n][j]
            #take symmetric edge
            Tej = T[ej[0]],T[ej[1]]
            if Tej[1]<Tej[0]:
                Tej = Tej[1],Tej[0]

            #find the intersection of the symmetric edge on the symmetric node
            #this position gives the position on the symmetric convex hull
            Tj = self.graph.incident_edges[Tn].index(Tej)



            right = "x%di%dj%d" % ( (Tn,Ti,Tj) if Ti<Tj else (Tn,Tj,Ti))

            equation = "%s = %s" % (left,right)

            yield (eq_name,equation)

    def compute_intersections(self):
        """Compute the intersection point of the unit sphere centered at each node with the incident edges"""

        #init the intersection points for each node
        self.avs = [{} for _ in range(len(self.graph.nodes))]

        #iterate over the incident edges on each node
        for i,av in enumerate(self.avs):
            for edge in self.graph.incident_edges[i]:
                j = edge[0] if edge[0]!=i else edge[1]
                p1,p2 = self.graph.nodes[i],self.graph.nodes[j]
                v = p2-p1
                v /= nla.norm(v)
                #add this intersection point
                av[edge]= v

    def compute_convex_hulls(self):
        """Compute the convex hulls of the intersection points at each node"""

        self.chs = []
        for i,av in enumerate(self.avs):
            ch = ConvexHull([av[edge] for edge in self.graph.incident_edges[i]],self.cos_merge)
            self.chs.append(ch)

    def generate_lp_indices_from_node_and_edge(self, i, edge):
        #find the index of the incident edge for current node
        k = self.graph.incident_edges[i].index( edge )
        #this edge index gives the intersection point self.chs[i].points[k]
        #find the incident edges in the convex hull to the point self.chs[i].points[k]
        point_edges = self.chs[i].point_edges[k]


        #indices have the format i<a>_<b>_<b> where
        #a is the index of the node and (b,c) is the edge in the convex hull of a
        eq_indices = ["%di%dj%d" % (i,l,m) for (l,m) in point_edges]

        return eq_indices

    def generate_lp_equations(self, edge):
        """Generates en equation for the LP"""
        i,j = edge
        i_idxs = self.generate_lp_indices_from_node_and_edge(i,edge)
        j_idxs = self.generate_lp_indices_from_node_and_edge(j,edge)
        i_eq = " + ".join(["x%s" % s for s in i_idxs])
        j_eq = " + ".join(["x%s" % s for s in j_idxs])

        equations = [i_eq + " = " + j_eq] #default equation is only one
        if self.regular:
            equations = [i_eq + " = q", j_eq + " = q"]
        equations = [i_eq,j_eq]
        return equations

    def generate_lp_variables(self,i,ch):
        return ["x%di%dj%d" % (i,k,l) for (k,l) in ch.edges ]

    def generate_IP(self):

        with open("lp.mod","wt") as f:

            quads_sum = []
            vars_to_display = []

            if self.regular:
                f.write("var q, integer, >= %d;" % self.min_subdivs)

            #generate equation variables for minimal number of points on each cell
            if self.less_subdivs:
                for k in range(len(self.graph.edges)):
                    f.write("var q%d, integer, >= %d;\n" % (k,self.min_subdivs))

            for i,ch in enumerate(self.chs):
                nn = len(ch.points)
                vs = self.generate_lp_variables(i,ch)
                vars_to_display.extend(vs)
                for var in vs:
                    #generate the definition of the variable
                    if self.less_subdivs and not ch.planar:
                        #test for less subdivisions
                        f.write("var %s, integer, >= %d;\n" % (var, 1 if nn>2 else self.min_subdivs) )
                    else:
                        #arcs in dangling nodes and articulations must be divided
                        #at least 4 times the rest of joints only 2
                        f.write("var %s, integer, >= %d;\n" % (var, 2 if nn>2 else self.min_subdivs) )

                    #EXPERIMENTAL
                    #arcs with minimal subdivision of 1
                    # if not ch.planar:
                    #     f.write("var %s, integer, >= %d;\n" % (var, 1 if nn>2 else 3) )
                    #END EXPERIMENTAL

                    #generate the terms in the objective sum
                    #dangling nodes are less expensive while the arcs in other
                    #nodes cost double
                    quads_sum.append("%s%s" % ("2*" if nn>1 else "",var))


            f.write("\nminimize quads_bound: " + " + ".join(quads_sum) + ";\n\n")


            for k,edge in enumerate(self.graph.edges):
                i,j = edge
                equations = self.generate_lp_equations(edge)
                if self.regular:
                    f.write("s.t. edge_%d_%d_compat_reg_%d: %s = q;\n" % (i,j,i,equations[0]) )
                    f.write("s.t. edge_%d_%d_compat_reg_%d: %s = q;\n" % (i,j,j,equations[1]) )
                else:
                    f.write("s.t. edge_%d_%d_compat_: %s = %s;\n" % (i,j,equations[0],equations[1]) )
                    if self.less_subdivs:
                        f.write("s.t. edge_%d_%d_compat_min_quads: %s = q%d;\n" % (i,j,equations[0],k) )

            if self.symmetric:
                for idx_T in range(len(self.symmetries)):
                    for i in range(len(self.graph.nodes)):
                        for data in self.generate_symmetric_equations(i,idx_T):
                            f.write("s.t. %s: %s;\n" % data)

            f.write("\nsolve;\n\n")

            f.write("printf: 'quads_bound %d\\n', quads_bound >> 'lp.sol'; \n")
            if self.regular:
                f.write("printf: 'quads_per_segment %d\\n', q >> 'lp.sol'; \n")
            for v in vars_to_display:
                f.write("printf: '%s %%d\\n', %s >> 'lp.sol'; \n" % (v,v))
            if self.less_subdivs:
                for k in range(len(self.graph.edges)):
                    f.write("printf: 'quads_per_segment%d %%d\\n', q%d >> 'lp.sol'; \n" % (k,k))

            f.write("end;\n")

    def read_solution(self):
        self.solution = {i: {} for i in range(len(self.chs))}
        self.quads_bound = None
        with open("lp.sol","rt") as f:
            exp = re.compile("x(\d+)i(\d+)j(\d+) (\d+)")
            exp2 = re.compile("quads_per_segment(\d+) (\d+)")
            #read the optimal number of quads
            m = re.match("quads_bound (\d+)",f.readline())
            self.quads_bound = int(m.groups()[0])

            if self.regular:
                m = re.match("quads_per_segment (\d+)",f.readline())
                self.quads_per_segment = int(m.groups()[0])

            #read the subdivisions per arc
            for line in f:
                m = exp.match(line)
                if m:
                    n,i,j,s = map(int, m.groups())
                    self.solution[n][(i,j)] = s
                else:
                    m = exp2.match(line)
                    _,cp = map(int, m.groups())
                    self.cross_profiles.add(cp)
                    #TODO: do something with the number of quads per segments

                #FIXME: the following patch is to guarantee that the fixed points in linked cells
                #are compatible, it's not a complete fix!!!
                #CORRECT FIX: detect the arcs that are aligned with a fixed point.
                #This means that, when there is a Voronoi vertex in one cell that is a fixed point
                #for a reflection symmetry through an incident edge (keeping that edge fixed)
                #and such that the projection of that Voronoi vertex onto the other cell of the
                #edge lies in the middle of an arc, that arc must be subdivided into an even
                #number of segments. This arc is identified as an arc-aligned-with-a-fixed-point.
                # if self.symmetries:
                #     self.solution[n][(i,j)] *= 2

    def solve_IP(self):
        open("lp.sol","w").close()
        # sp.call(["glpsol","--model","lp.mod"])
        sp.call(["glpsol","--model","lp.mod","--log","lp.log"])

        self.read_solution()

        os.remove("lp.mod")
        os.remove("lp.sol")

    def iterate_articulations_for_cells(self):
        """Returns a generator that iterates over the articulations returning a pair
        i,j where i is the current articulation and j is a node connected to i.
        If j is not None it means that the node j has its cells already computed."""
        art_idxs = set(self.graph.get_articulation_indices())
        while art_idxs:
            found = None
            for i in art_idxs:
                edge = self.graph.incident_edges[i][0]
                j = edge[0] if i!=edge[0] else edge[1]

                if self.cells[j]:
                    found=(i,j)
                    break

                edge = self.graph.incident_edges[i][1]
                j = edge[0] if i!=edge[0] else edge[1]

                if self.cells[j]:
                    found=(i,j)
                    break

            if not found:
                i = art_idxs.pop()
                found = (i,None)
            else:
                art_idxs.remove(i)

            yield found

    def create_cells(self):

        #create cells for joints first
        for idx,ch in enumerate(self.chs):
            #find point for current node
            # p = self.graph.nodes[idx]
            if self.graph.is_joint(idx):
                #save the new vertices valencies
                for facet in ch.facets:
                    self.vertex_valencies[2*len(facet)] += 1

                #find edges in the convex hull of the current joint
                for graph_edge,edges in zip(self.graph.incident_edges[idx],ch.point_edges):
                    cell = [] #new cell for this point in the convex hull

                    for edge in edges:
                        subdiv = self.solution[idx][edge] * self.quad_subdiv

                        n1,n2,phi = ch.edge_arc[edge]
                        #radius for current node
                        r = self.radii[idx]
                        #points in the cell
                        qs = [r*np.cos(t)*n1 + r*np.sin(t)*n2 for t in np.linspace(0,phi,subdiv+1)]

                        #save the new vertices valencies
                        #each of this vertices is counted twice so
                        #count only half on each cell
                        self.vertex_valencies[4] += (subdiv-1)*0.5

                        if not cell:
                            cell.extend(qs)
                        else:
                            #we have to check for the order of the arcs
                            #such that the cell is a closed polyline
                            if equal(cell[-1],qs[0]):
                                cell.extend(qs[1:])
                            elif equal(cell[0],qs[0]):
                                cell.reverse()
                                cell.extend(qs[1:])
                            elif equal(cell[-1],qs[-1]):
                                qs.reverse()
                                cell.extend(qs[1:])
                            elif equal(cell[0],qs[-1]):
                                cell.reverse()
                                qs.reverse()
                                cell.extend(qs[1:])
                            else:
                                raise ValueError("The cell is not a closed polyline!")

                    #for current node and current graph edge store the cell
                    self.node_cells[idx][graph_edge] = cell

        #create the cells for articulations
        for idx,ch in enumerate(self.chs):
            if self.graph.is_articulation(idx):
                self.create_articulation_cell(idx,ch)

        #create the cells for dangling nodes
        for idx,ch in enumerate(self.chs):
            if self.graph.is_dangling(idx):
                self.create_dangling_cell(idx,ch)

    def create_articulation_cell(self,idx,ch):
        subdiv = self.solution[idx][(0,1)] * self.quad_subdiv
        n1,n2,phi = ch.edge_arc[(0,1)]

        #check if this articulation has oposed edges
        if nla.norm(n1)<1e-7:
            k,l = self.graph.incident_edges[idx][0]
            if k==idx:
                k = l
            v = self.graph.nodes[idx]-self.graph.nodes[k]
            n1 = np.random.rand(3)
            n1 = np.cross(n1,v)
            n1 /= nla.norm(n1)
            n2 = np.cross(n1,v)
            n2 /= nla.norm(n2)

        r = self.radii[idx]
        qs = [r*np.cos(t)*n2 + r*np.sin(t)*n1 for t in np.linspace(0,phi,subdiv+1)]
        for graph_edge in self.graph.incident_edges[idx]:
            self.node_cells[idx][graph_edge] = qs
        #save the new vertices valencies
        self.vertex_valencies[4] += subdiv

    def create_dangling_cell(self,idx,_):
        subdiv = self.solution[idx][(0,0)] * self.quad_subdiv

        n = self.avs[idx].values()[0]

        #find the node connected with this dangling node
        graph_edge = self.graph.incident_edges[idx][0]
        j = graph_edge[0] if graph_edge[0]!=idx else graph_edge[1]

        #radius for current node
        r = self.radii[idx]

        n1 = None
        #if the connected node is also dangling
        #or we are not interested in symmetries
        if self.graph.is_dangling(j):
            #generate random normal direction
            n1 = np.random.rand(3)
            n1 = np.cross(n,n1)
            n1 /= nla.norm(n1)
            n2 = np.cross(n,n1)
            n2 /= nla.norm(n2)


            phi = 2*np.pi
            qs = [r*np.cos(t)*n1 + r*np.sin(t)*n2 for t in np.linspace(0,phi,subdiv+1)]

            #save the new vertices valencies
            self.vertex_valencies[3] += subdiv

            self.node_cells[idx][graph_edge] = qs
        else:
            ################################
            #generate a compatible normal
            # p = self.node_cells[j][graph_edge][0]
            # n1 = np.cross(np.cross(n,p),n)
            # n1 /= nla.norm(n1)
            ################################
            #project the cell onto this one
            qs = []
            for p in self.node_cells[j][graph_edge]:
                n1 = np.cross(np.cross(n,p),n)
                n1 /= nla.norm(n1)
                qs.append(n1)

            self.node_cells[idx][graph_edge] = [r*q for q in qs]

            #save the new vertices valencies
            self.vertex_valencies[3] += len(qs)-1

    def pair_cells(self,c1,c2,p1,p2):

        #Distance-based pairing
        c1 = [cell_point+p1 for cell_point in c1[:-1]]
        c2 = [cell_point+p2 for cell_point in c2[:-1]]
        n = len(c1)

        #Reflection-based pairing
        # n = P1-P2
        # n /= nal.norm(n)
        # lc1 = np.array([2*project_into_plane(p,n) for p in c1])-c1
        # lc2 = c2

        dc1 = co.deque(c1)
        min_i = 0
        min_d = sum(nla.norm(cp1-cp2) for cp1,cp2 in zip(dc1,c2))
        reverse = False

        for i in xrange(1,n):
            dc1.rotate(1)
            ###### Only need to try with the reversed cell if the rotation
            ###### order was taken into account for the ordering of the facets

            d = sum(nla.norm(cp1-cp2) for cp1,cp2 in zip(dc1,c2))
            if d < min_d:
                min_d = d
                min_i = i

        dc1 = co.deque(c1)
        dc1.reverse()
        for i in xrange(n):
            d = sum(nla.norm(cp1-cp2) for cp1,cp2 in zip(dc1,c2))
            if d < min_d:
                min_d = d
                min_i = i
                reverse = True
            dc1.rotate(1)

        idxs = co.deque(range(n))
        if reverse: idxs.reverse()
        idxs.rotate(min_i)

        # if ORDER_OF_CELLS==REVERSE: rev=not rev
        # res = zip( np.roll(c1,min_i,axis=0), c2[::-1] if rev else c2)
        # if ORDER_OF_CELLS==RANDOM:
        #     n = c2.shape[0]
        #     res = zip(c1,c2[random.sample(range(n),n)])
        return idxs

    def create_links(self):
        self.links = {} #store for each edge the linking details
        for edge in self.graph.edges:
            i,j =  edge
            p1 = self.graph.nodes[i]
            p2 = self.graph.nodes[j]

            cell1 = self.node_cells[i][edge]
            cell2 = self.node_cells[j][edge]

            self.links[edge] = self.pair_cells(cell1,cell2,p1,p2)
            # print edge,self.links[edge]

    def get_axel_visualization(self):
        vis = visual.get_axel_visualization()
        self.draw(vis)
        return vis

    def draw(self,vis):
        g = self.graph
        s = self

        vis.add_points(g.nodes,name="extremities",color=visual.black)
        for i,j in g.edges:
            vis.add_polyline([g.nodes[i],g.nodes[j]],color=visual.red,name="skeleton")
        for i,av in enumerate(s.avs):
            ch,p = s.chs[i],g.nodes[i]
            vis.add_points([v+p for v in av.values()],name="intersections")
            if len(av)>1:
                for j,k in ch.edges:
                    vis.add_polyline([av[g.incident_edges[i][j]]+p,av[g.incident_edges[i][k]]+p],color=visual.black,name="edges")


            if len(av)>1:
                for e in ch.edges:
                    n1,n2,phi = ch.edge_arc[e]
                    vis.add_polyline([p + np.cos(t)*n1 + np.sin(t)*n2 for t in np.linspace(0,phi,int(phi/(2*np.pi)*50))], color=visual.yellow, name="arcs")
                    vis.add_points([p+n1,p+np.cos(phi)*n1+np.sin(phi)*n2],color=visual.blue,name="voronoi_sites")
                    n1,n2 = ch.edge_normals[e]

        #plot cells
        for i,edges in enumerate(g.incident_edges):
            p = g.nodes[i]
            for edge in edges:
                cname = ("cell %d,%d,%d" % (i,edge[0],edge[1])) if self.split_output else "cells"
                vis.add_polyline([p+q for q in s.node_cells[i][edge]], color=visual.blue,name=cname )


        total_quads = 0
        for edge in g.edges:

            i,j = edge
            cell1 = s.node_cells[i][edge][:-1]
            cell2 = s.node_cells[j][edge][:-1]
            p1 = g.nodes[i]
            p2 = g.nodes[j]
            n = len(cell1)

            total_quads += n

            idxs = s.links[edge]
            dc1 = [cell1[idx] for idx in idxs]

            if self.palette:
                k = int(np.random.rand()*len(self.palette))
                for i in range(n):
                    i2 = (i+1)%n
                    ps = [p1+dc1[i],p2+cell2[i],p1+dc1[i2],p2+cell2[i2]]
                    qname = ("quads %d,%d" % edge) + (" %d" % i)
                    vis.add_mesh(ps,[[0,1,3,2]],color=self.palette[(i+k) % len(self.palette)])
            else:
                qname = ("quads %d,%d" % edge) if self.split_output else "quads"

                ps = [p1 + q for q in dc1] + [p2 + q for q in cell2]
                quads = [[l,(l+1)%n,(l+1)%n+n,l+n] for l in range(n)]

                qname = ("quads %d,%d" % edge) if self.split_output else "quads"
                vis.add_mesh(ps,quads,color=visual.dark_yellow,name=qname)


            for p,q in zip(dc1,cell2):
                mname = ("mesh lines %d,%d" % edge) if self.split_output else "mesh lines"
                vis.add_polyline([p1+p,p2+q],color=visual.cyan,name=mname)

        print "TOTAL QUADS IN VISUALIZATION",total_quads
        return vis

    def _file_name(self,path):
        res = path+"/"+self.name
        if self.symmetric:
            res += "_sym"
        if self.regular:
            res += "_reg"
        return res

    def save_log_data(self,fname):
        articulations = len(self.graph.get_articulation_indices())
        danglings = len(self.graph.get_dangling_indices())
        joints = len(self.graph.nodes) - articulations - danglings
        quads = 0
        for k in sorted(self.vertex_valencies.keys()):
            if k==3:
                quads += 2*self.vertex_valencies[k]
            else:
                quads += k*self.vertex_valencies[k]
        quads /= 4
        with open(fname,"wt") as f:
            f.write("TOTAL TIME: %f\nLP TIME: %f\nCONVEX HULL TIME: %f\nOTHER TIME: %f\n" % (self.running_time,self.lp_time,self.ch_time,self.running_time-self.lp_time-self.ch_time))
            f.write("Skeleton summary:\n")
            f.write("Total Nodes %d\nTotal Edges %d\n" % (len(self.graph.nodes),len(self.graph.edges)) )
            f.write("Articulations %d\n" % articulations)
            f.write("Dangling nodes %d\n" % danglings)
            f.write("Joints %d\n" % joints)
            f.write("Scaffold summary:\n")
            f.write("Total quads: %d\n" % quads )
            f.write("Cross profiles: %s\n" % " ".join(map(str,sorted(self.cross_profiles))))
            f.write("Number of vertices by valency:\n")
            for k in sorted(self.vertex_valencies.keys()):
                if self.vertex_valencies[k]: f.write(" %d: %d\n" % (k,self.vertex_valencies[k]) )

    def save_lp_log_data(self, fname):
        import shutil
        shutil.copy2("./lp.log",fname)

    def save_axel_visualization(self,path="./"):
        vis = self.get_axel_visualization()
        vis.output_file = self._file_name(path)+".axl"
        vis.save()
        return vis

    def save_all(self,path="./"):
        self.save_log_data(self._file_name(path)+".log")
        self.save_lp_log_data(self._file_name(path)+"_lp.log")
        return self.save_axel_visualization(path)

    def read_symmetries(self,fname):
        self.set_symmetric(True)
        self.clear_symmetries()
        with open(fname,"rt") as f:
            for line in f:
                if line[0] == "#":
                    continue
                if line:
                    T = map(int,line.split(","))
                    self.add_symmetry(T)

if __name__=="__main__":


    g = Graph()
    g.read_from_graph_file("/user/afuentes/home/Work/Convolution/code/python/skels_data/6ring.graph")

    s = Scaffolder(g)
    s.name="6ring"
    s.less_subdivs = True
    # s.min_subdivs = 8
    # s.cos_merge=1.5
    # s.quad_subdiv = 2

    # s.read_symmetries("skels_data/sym_problem_graph.sym")

    # s.set_regular(True)
    # s.set_symmetric(True)

    # s.set_all_radius(0.1)
    # s.set_radii([0.4,1.0,0.5,0.6,1.0,1.0,0.2])
    # s.set_node_radius(0,2.0)
    # s.set_node_radius(2,2.0)
    # s.set_node_radius(4,2.0)

    # s.set_node_radius(12,1.5)
    # s.set_node_radius(13,1.5)
    # s.set_node_radius(14,1.5)


    #construct the scaffold

    s.compute_scaffold()
    s.get_axel_visualization().show()
    # hex_scaffold_vis(s).show()

    #show the scaffold

    #s.set_palette(pastel_palette)
    # s.save_log_data("scaffold.log")
    # s.get_axel_visualization().show()
    # s.save_all(path="/user/afuentes/home/Work/Writing/SPM 2018/axls/less_quads/")
    # s.save_all(path="/user/afuentes/home/Work/Axel models").show()
