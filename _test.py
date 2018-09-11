import scaffolder as sc
import numpy as np
import math

from _math import *
import biarcs
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


    a = [1.0,1.0]
    b = [2.0,2.0]
    c = [1.0,1.0]
    th = [0.0,0.0]
    th = [0.0*np.pi,0.5*np.pi]

    arc_skeleton = sk.Arc(C,u,v,r,phi)
    field = fl.ArcField(1.0,arc_skeleton,a,b,c,th)

    return g,field


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


    a = np.array([1.0,1.0])
    b = np.array([2.0,2.0])
    c = np.array([1.0,1.0])
    th = np.array([np.pi*0.5,np.pi])
    th = np.array([0.0,0.0])
    th = np.array([np.pi*0.5,np.pi*0.5])
    def get_field(i,j):
        A = g.nodes[i]
        B = g.nodes[j]
        n = normalize(np.cross(np.array([1.0,1.0,0.0]),B-A))

        segment_skeleton = sk.Segment.make_segment(A,B,n)
        return fl.SegmentField(1.0,segment_skeleton,a,b,c,th)


    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 16
    scaff.compute_scaffold()

    fs = [get_field(*e) for e in g.edges]

    field = fl.MultiField(fs)

    return g,field

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

    arc_skeleton = sk.Arc(C,u,v,r,phi)
    fs.append(fl.ArcField(1.0,arc_skeleton,[1.0,1.0],[2.0,1.0],[1.0,1.0]))


    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 30
    scaff.compute_scaffold()



    field = fl.MultiField(fs)
    return g,field

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

    L = np.array([4278336.0/1722133.0,3788856.0/708271.0,0.0])
    M = np.array([6260761.0/1288993.0,8005913.0/1419541.0,0.0])
    N = np.array([5741525.0/841557.0,4636157.0/1091265.0,0.0])

    ds = [
        (A,C,J), #legs
        (B,A,F),#baby
        (L,B,G),#baby neck
        (E,N,G),#mother neck
        (D,E,H),#back
        (D,A,K),#middle
        # (C,D,I),#floor
    ]

    a = [1.0,1.0]

    bb = [
        [1.0,1.8],#legs
        [1.2,1.2],#baby
        [0.8,0.8],#baby neck
        [0.8,0.8],#mother neck
        [1.8,1.5],#back
        [2.5,1.0] #middle
        # [1.2,1.2],#floor
    ]

    cc = [
        [1.0,2.8],#legs
        [1.5,1.2],#baby
        [0.8,0.8],#baby neck
        [0.8,0.8],#mother neck
        [1.6,1.5],#back
        [2.3,1.0] #middle
        # [3.0,3.0],#floor
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

        print r

        arcs.append((Ce,u,v,r,phi))

    g = gr.Graph()
    fs = []
    #floor part
    g.add_node(D)
    g.add_node(C)
    g.add_edge(0,1)
    fs.append(fl.SegmentField(1.0,sk.Segment.make_segment(D,C,z,),a,[3.0,3.0],[1.2,1.2]))

    #node for the heads
    g.add_node(M) #idnex 2

    for arc,b,c in zip(arcs,bb,cc):
        ni = [g.add_node(n) for n in arc_to_nodes(*arc)]
        g.add_edge(ni[0],ni[1])
        g.add_edge(ni[1],ni[2])
        g.add_edge(ni[2],ni[3])
        g.add_arc(ni)
        C,u,v,r,phi = arc
        fs.append(fl.ArcField(1.0,sk.Arc(C,u,v,r,phi),a,b,c))

    #baby head
    g.add_edge(2,g.find_node_from_point(L))
    fs.append(fl.SegmentField(1.0,sk.Segment.make_segment(M,L,z),[0.5,1.0],[1.1,1.1],[1.3,1.3]))

    #mother head
    g.add_edge(2,g.find_node_from_point(N))
    fs.append(fl.SegmentField(1.0,sk.Segment.make_segment(M,N,z),[0.5,1.0],[1.1,1.1],[1.3,1.3]))

    field = fl.MultiField(fs)

    return g,field

def knot():
    #Knot curve
    gamma = lambda t: np.array([
            -10*np.cos(t)-2*np.cos(5*t)+15*math.sin(2*t),
            -15*np.cos(2*t)+10*math.sin(t)-2*math.sin(5*t),
            10*np.cos(3*t)
            ])
    def gammat(t):
        v = np.array([
            #-10*np.cos(t) -2* np.cos(5*t)+ 15*math.sin(2*t),
            10*math.sin(t) +10*math.sin(5*t)+ 30*np.cos(2*t),
            #-15*np.cos(2*t)+10*math.sin(t)-2*math.sin(5*t),
            30*math.sin(2*t)+10*np.cos(t)-10*np.cos(5*t),
            #10*np.cos(3*t)
            -30*math.sin(3*t)])
        v /= nla.norm(v)
        return v
    t0,t1 = .0,2*np.pi

    #compute biarcs approximation
    num_samples = 20
    aa = biarcs.biarcs_from_curve(gamma,gammat,t0,t1,num_samples)

    g = gr.Graph()
    fs = []
    a = np.array([1.0,1.0])
    b = np.array([1.0,1.0])
    c = np.array([1.0,1.0])
    el = 0.5/float(len(aa))
    for i,(C,u,v,r,phi) in enumerate(aa):
        ni = [g.add_node(n) for n in arc_to_nodes(C,u,v,r,phi)]
        g.add_edge(ni[0],ni[1])
        g.add_edge(ni[1],ni[2])
        g.add_edge(ni[2],ni[3])
        g.add_arc(ni)
        fs.append(fl.ArcField(1.0,sk.Arc(C,u,v,r,phi),a,b-(el*i),c+(el*i)))
    field = fl.MultiField(fs)

    return g,field

def compute(g,field):
    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 10
    mesher = ms.Mesher(scaff,field)
    mesher.quads_num = 8
    # mesher.split_output = True
    mesher.parallel_ray_shooting = False

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
    # fertility()
    # compute(*fertility())
    # compute(*combined_scaff())
    # compute(*segment_scaff())
    # compute(*arc_scaff())
    compute(*knot())
