import numpy as np
import numpy.linalg as nla
import re
import collections

from ._math import *

class Graph(object):
    """Graph input to scaffolder"""
    def __init__(self):
        super(Graph, self).__init__()
        self.nodes = []
        self.edges = []
        self.incident_edges = []
        self.data = collections.defaultdict(lambda: [])

    def add_node(self,point):
        point = np.array(point)
        idx = self.find_node_from_point(point)
        if idx==-1:
            idx = len(self.nodes)
            self.nodes.append(point)
            self.incident_edges.append([])
        return idx

    def add_edge(self,i,j):
        edge = make_edge(i,j)
        if edge in self.edges:
            return
        self.edges.append(edge)
        assert i<len(self.nodes),"Wrong edge {},{} -- {}>={}".format(i,j,i,len(self.nodes))
        assert j<len(self.nodes),"Wrong edge {},{} -- {}>={}".format(i,j,j,len(self.nodes))
        self.incident_edges[i].append(edge)
        self.incident_edges[j].append(edge)
        return edge

    def is_joint(self, i):
        return len(self.incident_edges[i])>2

    def is_dangling(self, i):
        return len(self.incident_edges[i])==1

    def is_articulation(self, i):
        return len(self.incident_edges[i])==2

    def save_to_skel_file(self,fname):
        with open(fname,"wt") as f:
            for i,j in self.edges:
                node1 = self.nodes[i]
                node2 = self.nodes[j]
                f.write("%f %f %f    %f %f %f    1\n" % tuple(list(node1)+list(node2)) )

    def save_to_graph_file(self,fname):
        with open(fname,"wt") as f:
            f.write("nodes\n")
            for node in self.nodes:
                f.write("%f %f %f\n" % tuple(node) )
            f.write("edges\n")
            for i,j in self.edges:
                f.write("%d %d \n" % (i,j) )

            f.write("radii\n")
            for r in self.data["radii"]:
                f.write("{}\n".format(r))

    def get_articulation_indices(self):
        return [i for i in range(len(self.nodes)) if self.is_articulation(i)]

    def get_dangling_indices(self):
        return list(filter(self.is_dangling,range(len(self.nodes)))) # [i for i in range(len(self.nodes)) if self.is_dangling(i)]

    def find_node_from_point(self, p):
        for i,q in enumerate(self.nodes):
            if nla.norm(p-q)<1e-10:
                return i
        return -1

    def read_from_skel_file(self,fname):
        self.nodes = []
        self.edges = []
        self.incident_edges = []
        spc = re.compile("\s+")
        init_spc = re.compile("^\s+")
        with open(fname,"rt") as f:
            for line in f:
                line = init_spc.sub("",line)
                line = spc.sub(" ",line)
                if line:
                    x0,y0,z0,x1,y1,z1,r = list(map(float,line.split()))
                    p0 = np.array([x0,y0,z0])
                    p1 = np.array([x1,y1,z1])
                    i = self.add_node(p0)
                    j = self.add_node(p1)
                    if j<i: i,j = j,i
                    self.add_edge(i,j)

    def read_from_graph_file(self,fname):
        self.nodes = []
        self.edges = []
        self.incident_edges = []

        reading = None
        with open(fname,"rt") as f:
            node_remap = []
            for line in f:
                line = line.strip()
                if not line or line[0]=="#": continue
                if line.isalpha():
                    reading=line
                    read_nodes=0
                elif reading=="nodes":
                    node = np.fromstring(line,sep=" ")
                    i = self.add_node(node)
                    node_remap.append(i)
                elif reading=="edges":
                    i,j = list(map(int,line.split()))
                    i = node_remap[i]
                    j = node_remap[j]
                    self.add_edge(i,j)
                else:
                    val = float(line)
                    #add value only if the node is not repeated
                    if node_remap[read_nodes]>=len(self.data[reading]):
                        self.data[reading].append(val)
                    else:
                        print("no data for {}, due to {}->{}".format(read_nodes,read_nodes,node_remap[read_nodes]))
                    read_nodes+=1

