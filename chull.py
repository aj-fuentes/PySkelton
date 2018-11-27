import pyhull
import numpy as np
import numpy.linalg as nla

class ConvexHull(object):
    """ConvexHull"""

    def __init__(self, points,cos_merge=0.999):
        super(ConvexHull, self).__init__()
        self.points = np.array(points)
        self.edges = None
        self.planar = False #indicated whether this convex hull is planar
        self.point_edges = None #edges incident to a point
        self.normals = None #face normals in the convex hull (outward pointing)
        self.edge_normals = None #normals of the faces adjacent to the edge
        self.edge_arc_midpoint = None #midpoint of the arc perpendicular to an edge
        self.edge_arc = None #arc perpendicular to edge (n1,n2,angle)

        self.cos_merge = cos_merge #defines the angle for two facets to be considered coplanar

    def get_point_from_graph_edge(self,edge):
        return self.graph_edges.index(edge)


    def compute_data(self):
        """Compute the convex hull and associated data"""

        self.edges = set()
        self.point_edges = [[] for i in range(len(self.points))]
        self.edge_normals = {}

        ps = self.points

        if len(ps)==1:
            self.edges.add( (0,0) ) #phantom edge
            self.point_edges[0].append((0,0)) #only one edge
            self.edge_normals= {(0,0):[np.zeros(3),np.zeros(3)]}
        elif len(ps)==2:
            self.edges.add( (0,1) ) #unique edge
            self.point_edges[0].append( (0,1) ) #only one edge
            self.point_edges[1].append( (0,1) ) #only one edge

            #compute the normal of the arc
            n1 = np.cross(ps[0],ps[1])
            if nla.norm(n1)>1e-7:
                n1 /= nla.norm(n1)

            self.edge_normals = {(0,1):[n1,-n1]}

            #compute second normal of the arc
            n2 = ps[0]+ps[1]
            if nla.norm(n2)>1e-7:
                n2 /= nla.norm(n2)

            self.edge_arc = {(0,1):(n1,n2,2*np.pi)}
        else:
            data = ['']
            try:
                data = pyhull.qconvex('i FN Fn n -A%1.5f' % self.cos_merge,ps)
            except:
                data = ['']
            if not data[0]:
                #try the planar convex hull projecting in XY
                self.planar = True
                try:
                    data = pyhull.qconvex('i FN Fn n Qb0:0B0:0 -A%1.5f' % self.cos_merge,ps)
                except:
                    data = ['']
                if not data[0]:
                    #try the planar convex hull projecting in XZ
                    try:
                        data = pyhull.qconvex('i FN Fn n Qb1:0B1:0 -A%1.5f' % self.cos_merge,ps)
                    except:
                        data = ['']
                    if not data[0]:
                        #it should never get to this point!!!
                        #try the planar convex hull projecting in YZ
                        data = pyhull.qconvex('i FN Fn n Qb2:0B2:0 -A%1.5f' % self.cos_merge,ps)
            self.parse_data(data)

    def parse_data(self,data):

        self.number_of_facets = number_of_facets = int(data[0])
        ps = self.points

        self.facets = [ map(int,f.split()) for f in data[1 : 1+number_of_facets] ]
        # print "FACETS"
        # for line in data[1 : 1+number_of_facets]:
        #     print line

        #neighboring facets to a point
        self.point_neighbors = [ map(int,f.split())[1:] for f in data[2+number_of_facets : 2+number_of_facets+ps.shape[0]] ]
        # print "POINT NEIGHBORS"
        # for line in data[2+number_of_facets : 2+number_of_facets+ps.shape[0]]:
        #     print line

        #neighboring facets to a point
        self.facet_neighbors = [ map(int,f.split())[1:] for f in data[3+number_of_facets+ps.shape[0]: 3+2*number_of_facets+ps.shape[0]] ]
        # print "FACET NEIGHBORS"
        # for line in data[3+number_of_facets+ps.shape[0]: 3+number_of_facets+2*ps.shape[0]]:
        #     print line

        #outward normals of the facets
        self.normals = [ np.array(map(float,n0.split())[:-1]) for n0 in data[5+2*number_of_facets+ps.shape[0]:]  ]
        # print "FACET NORMALS"
        # for line in data[5+number_of_facets+2*ps.shape[0]:]:
        #     print line

    def process_data(self):

        if len(self.points)<3:
            return

        ps = self.points

        if self.planar:
            n = np.cross(self.points[1]-self.points[0],self.points[2]-self.points[0])
            n /= nla.norm(n)
            self.normals = [n for _ in self.facets]

        #init the arcs for the edges
        self.edge_arc = {}

        def compute_edge_data(i,f,f2):
            """Computes the edges incidents to node i from the two adjacent facets f and f2"""

            if self.planar: #planar convex hull
                j = self.facets[f][0]
                if j==i: j = self.facets[f][1]
            else:
                #common points between facets define the edge
                common_points = set(self.facets[f]).intersection(self.facets[f2])
                #i must be in the common points of the facets
                common_points.remove(i)
                #the other point is the other extremity of the edge
                j = common_points.pop()

            #create the edge
            edge = (i,j) if i<j else (j,i)
            #add to list of edges
            self.edges.add(edge) #some edges might be already present but this is a set!

            #add the edge to the list of adjacent edges to this point
            self.point_edges[i].append(edge)


            #save the two normals of the facets common to the edge
            if self.planar:
                n = self.normals[f]
                self.edge_normals[edge] = (n,-n)
            else:
                self.edge_normals[edge] = (self.normals[f],self.normals[f2])


            #compute the arc of this edge
            if self.planar:
                n1 = self.edge_normals[edge][0]
                n2 = self.points[i] + self.points[j]
                n2 = np.cross(np.cross(n1,n2),n1)
                n2 /= nla.norm(n2)
                #barycenter of one triangle in the facet
                b = (self.points[0]+self.points[1]+self.points[2])/3.0
                #vector from midpoint of edge to barycenter
                b -= (self.points[i] + self.points[j])/2.0
                b /= nla.norm(b)
                if np.dot(n2,b)>0: n2 = -n2
                self.edge_arc[edge] = (n1,n2,np.pi)
            else:
                #the arc starts at normal1 and moves in the plane of normal1,normal2 towards normal2
                normal1,normal2 = self.edge_normals[edge]

                n1 = normal1 #first axis of the arc
                n2 = np.cross(np.cross(normal1,normal2),normal1)
                n2 /= nla.norm(n2) #second axis of the arc

                phi = np.arccos(np.dot(normal1,normal2)) #angle of the arc

                self.edge_arc[edge] = (n1,n2,phi) #save the arc for this edge

        for i in range(ps.shape[0]):

            #order the facets neighboring a point such that two consecutive facets share an edge
            facets_bag = set(self.point_neighbors[i])
            first_facet = facets_bag.pop()
            f = first_facet
            while facets_bag:
                facets_with_common_edge = set(self.facet_neighbors[f]).intersection(facets_bag)
                #take one of the facets with a common edge with the current one
                #notice that there must be at most two
                f2 = facets_with_common_edge.pop()
                compute_edge_data(i,f,f2)

                #remove new facet from the bag
                facets_bag.remove(f2)
                #continue with the next facet
                f = f2
                #add first facet if the bag is empty
                #this is to catch the common edge between the first facet and the last one
            #we must catch the last edge between f and fisrt_facet
            compute_edge_data(i,f,first_facet)


