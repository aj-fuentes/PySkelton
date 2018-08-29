import scaffolder as sc
import numpy as np
from _math import *
import graph as gr
import mesher as ms
import skeleton as sk
import field as fl

def arc_scaff():

    C = np.array([0.4,-3.0,10.0])
    r = 4.0
    phi = np.pi*0.6
    u = normalize(np.array([1.0,1.0,0.0]))
    v = normalize(np.array([1.0,-1.0,0.0]))
    arc = C,u,v,r,phi

    g = gr.Graph()

    nodes = arc_to_nodes(C,u,v,r,phi)

    for n in nodes:
        g.add_node(n)
    for i in range(3):
        g.add_edge(i,i+1)

    g.add_arc([0,1,2,3])

    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 16
    scaff.compute_scaffold()

    arc_skeleton = sk.Arc(C,u,v,r,phi,[1.0,1.0],[2.0,1.0],[1.0,1.0])
    field = fl.ArcField(1.0,arc_skeleton)

    mesher = ms.Mesher(scaff,field)
    # mesher.quads_num=1

    vis = scaff.get_axel_visualization()

    vis.add_polyline(arc_points(arc),name="arc",color="black")

    mesher.draw(vis)

    vis.show()


def segment_scaff():

    C = np.array([0.4,-3.0,10.0])
    r = 4.0
    phi = np.pi*0.6
    u = normalize(np.array([1.0,1.0,0.0]))
    v = normalize(np.array([1.0,-1.0,0.0]))

    nodes = arc_to_nodes(C,u,v,r,phi)

    g = gr.Graph()
    for n in nodes:
        g.add_node(n)
    for i in range(3):
        g.add_edge(i,i+1)


    def get_field(i,j):
        A = g.nodes[i]
        B = g.nodes[j]
        n = normalize(np.cross(np.array([1.0,1.0,0.0]),B-A))

        segment_skeleton = sk.Segment.make_segment(A,B,n)
        return fl.SegmentField(1.0,segment_skeleton)


    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 16
    scaff.compute_scaffold()

    fs = [get_field(*e) for e in g.edges]

    field = fl.MultiField(fs)

    vis = scaff.get_axel_visualization()

    mesher = ms.Mesher(scaff,field)
    mesher.draw(vis)

    vis.show()

def combined_scaff():

    C = np.array([0.4,-3.0,10.0])
    r = 4.0
    phi = np.pi*0.6
    u = normalize(np.array([1.0,1.0,0.0]))
    v = normalize(np.array([1.0,-1.0,0.0]))

    nodes = arc_to_nodes(C,u,v,r,phi)

    g = gr.Graph()
    for n in nodes:
        g.add_node(n)
    for i in range(3):
        g.add_edge(i,i+1)

    g.add_arc([0,1,2,3])

    i = g.add_node(np.array([1.0,2.0,8.0]))
    g.add_edge(i,0)

    def get_field(i,j):
        A = g.nodes[i]
        B = g.nodes[j]
        n = normalize(np.cross(np.array([1.0,1.0,0.0]),B-A))

        segment_skeleton = sk.Segment.make_segment(A,B,n)
        return fl.SegmentField(1.0,segment_skeleton)


    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 16
    scaff.compute_scaffold()

    fs = [get_field(*e) for e in g.edges]

    arc_skeleton = sk.Arc(C,u,v,r,phi,[1.0,1.0],[2.0,1.0],[1.0,1.0])
    fs.append(fl.ArcField(1.0,arc_skeleton))


    field = fl.MultiField(fs)

    vis = scaff.get_axel_visualization()

    mesher = ms.Mesher(scaff,field)
    mesher.draw(vis)

    vis.show()


if __name__=="__main__":
    # arc_scaff()
    # segment_scaff()
    combined_scaff()
