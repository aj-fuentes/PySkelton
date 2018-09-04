import scaffolder as sc
import numpy as np
from _math import *
import graph as gr
import mesher as ms
import skeleton as sk
import field as fl
import visualization as visual

import cProfile
import timeit
import vmprof
import pstats

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


    fs = [get_field(*e) for e in g.edges]

    arc_skeleton = sk.Arc(C,u,v,r,phi,[1.0,1.0],[2.0,1.0],[1.0,1.0])
    fs.append(fl.ArcField(1.0,arc_skeleton))


    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 30
    scaff.compute_scaffold()



    field = fl.MultiField(fs)

    vis = scaff.get_axel_visualization()

    mesher = ms.Mesher(scaff,field)
    mesher.quads_num = 40
    # mesher.parallel_ray_shooting = False

    # cProfile.runctx("mesher.compute()",globals(),locals())

    mesher.draw(vis)


    vis.show()

def dragon():
    g = gr.Graph()
    g.read_from_graph_file("/user/afuentes/home/Work/Models/Zanni/SkelExemple/Dragon.graph")
    scaff = sc.Scaffolder(g)
    scaff.get_axel_visualization().show()

def fertility():
    A = np.array([0.0,0.0,0.0])
    B = np.array([49.0/50.0,37.0/10.0,0.0])
    C = np.array([-6.0,-6.0,0.0])
    D = np.array([263.0/50.0,-311.0/50.0,0.0])
    E = np.array([184.0/25.0,21.0/10,0.0])
    F = np.array([4973826.0/1541477,412099.0/366260,0.0])
    G = np.array([22263.0/5500.0,1508.0/625,0.0])
    H = np.array([5919635.0/2121837.0,-5537309.0/4726667.0,0.0])
    # I = np.array([-205011.0/7927250.0,45596081.0/3963625.0,0.0])
    J = np.array([-31.0/100.0,-569.0/100.0,0.0])
    K = np.array([-5599672.0/1981557.0,-5201743.0/673467.0,0.0])

    ds = [
        (B,A,F),#baby
        (E,B,G),#head
        (D,E,H),#back
        (D,A,K),#middle
        # (C,D,I),#floor
        (A,C,J) #legs
    ]

    bb = [
        [1.2,1.2],#baby
        [1.2,1.2],#head
        [1.8,1.5],#back
        [2.5,1.0],#middle
        # [1.2,1.2],#floor
        [1.0,1.8] #legs
    ]

    cc = [
        [1.5,1.2],#baby
        [0.9,0.9],#head
        [1.6,1.5],#back
        [2.3,1.0],#middle
        # [3.0,3.0],#floor
        [1.0,2.8] #legs
    ]


    z = np.array([0.0,0.0,1.0])

    arcs = []

    for d in ds:
        A0,A1,Ce = d

        u = normalize(A0-Ce)
        r = norm(A0-Ce)
        v = np.cross(z,u)
        w = normalize(A1-Ce)
        phi = np.arccos(np.dot(w,u))

        arcs.append((Ce,u,v,r,phi))

    g = gr.Graph()
    fs = []
    a = [1.0,1.0]
    for arc,b,c in zip(arcs,bb,cc):
        ni = [g.add_node(n) for n in arc_to_nodes(*arc)]
        g.add_edge(ni[0],ni[1])
        g.add_edge(ni[1],ni[2])
        g.add_edge(ni[2],ni[3])
        g.add_arc(ni)
        C,u,v,r,phi = arc
        fs.append(fl.ArcField(1.0,sk.Arc(C,u,v,r,phi,a,b,c)))

    #floor part
    g.add_edge(7,14)
    fs.append(fl.SegmentField(1.0,sk.Segment.make_segment(g.nodes[7],g.nodes[14],z,a,[3.0,3.0],[1.2,1.2])))

    field = fl.MultiField(fs)

    # vis = visual.VisualizationAxel()

    # for i,j in g.edges:
    #     vis.add_polyline([g.nodes[i],g.nodes[j]],name="skel",color="red")

    # vis.show()

    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 20
    mesher = ms.Mesher(scaff,field)
    mesher.quads_num = 50
    mesher.split_output = True
    # mesher.parallel_ray_shooting = False

    scaff.compute_scaffold()
    vis = scaff.get_axel_visualization()

    s = timeit.default_timer()

    # pr = cProfile.Profile()
    # pr.enable()
    mesher.compute()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats('cumulative')
    # ps.print_stats()

    e = timeit.default_timer()
    print "time={}".format(e-s)

    mesher.draw(vis)
    vis.show()


if __name__=="__main__":
    # arc_scaff()
    # segment_scaff()
    # combined_scaff()
    # dragon()
    fertility()
