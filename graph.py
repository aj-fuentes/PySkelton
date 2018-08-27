import numpy as np
import numpy.linalg as nla
import re


class Graph(object):
    """Graph input to scaffolder"""
    def __init__(self):
        super(Graph, self).__init__()
        self.nodes = []
        self.edges = []
        self.incident_edges = []

    def add_node(self,point):
        idx = self.find_node_from_point(point)
        if idx==-1:
            idx = len(self.nodes)
            self.nodes.append(point)
            self.incident_edges.append([])
        return idx

    def add_edge(self,i,j):
        edge = (i,j) if i<j else (j,i)
        self.edges.append(edge)
        self.incident_edges[i].append(edge)
        self.incident_edges[j].append(edge)

    def is_joint(self, i):
        return len(self.incident_edges[i])>2

    def is_dangling(self, i):
        return len(self.incident_edges[i])<2

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

    def get_articulation_indices(self):
        return [i for i in range(len(self.nodes)) if self.is_articulation(i)]

    def get_dangling_indices(self):
        return [i for i in range(len(self.nodes)) if self.is_dangling(i)]

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
                    x0,y0,z0,x1,y1,z1,r = map(float,line.split())
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
        spc = re.compile("\s+")
        init_spc = re.compile("^\s+")
        rnodes = False
        redges = False
        with open(fname,"rt") as f:
            for line in f:
                line = init_spc.sub("",line)
                line = spc.sub(" ",line)
                if line:
                    if line[:5]=="nodes":
                        rnodes = True
                        redges = not rnodes
                    elif line[:5]=="edges":
                        redges = True
                        rnodes = not redges
                    else:
                        if rnodes:
                            x,y,z = map(float,line.split())
                            p = np.array([x,y,z])
                            ni = self.add_node(p)
                            # print "node %d %f %f %f" % (ni,x,y,z)
                        elif redges:
                            i,j = map(int,line.split())
                            if j<i: i,j = j,i
                            # print "edge %d %d" % (i,j)
                            self.add_edge(i,j)
