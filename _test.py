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
import scipy.optimize as sop

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
    pieces = [(arc_skeleton,[0,1,2,3])]

    return g,field,pieces

def segment_scaff():

    A = np.array([5.0,0.0,0.0])
    B = np.array([3.0,4.0,0.0])
    C = np.array([0.0,0.0,0.0])

    g = gr.Graph()
    g.add_node(A)
    g.add_node(B)
    g.add_node(C)
    g.add_edge(0,1)
    g.add_edge(0,2)
    g.add_edge(1,2)


    fs = []
    pieces = []

    n = np.array([0.0,0.0,1.0])

    b = np.array([1.0,1.0])
    c = np.array([1.0,1.0])
    th = np.array([0.0,0.0])

    seg = sk.Segment.make_segment(A,B,n)
    fs.append(fl.SegmentField(1.0,seg,b=b,c=c,th=th))
    pieces.append((seg,[0,1]))

    seg = sk.Segment.make_segment(A,C,n)
    fs.append(fl.SegmentField(1.0,seg,b=b,c=c,th=th))
    pieces.append((seg,[0,2]))

    seg = sk.Segment.make_segment(B,C,n)
    fs.append(fl.SegmentField(1.0,seg,b=b,c=c,th=th))
    pieces.append((seg,[1,2]))

    field = fl.MultiField(fs)

    return g,field,pieces

def combined_scaff2():

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

    i = g.add_node(C)
    g.add_edge(i,0)

    j = g.add_node(C)
    g.add_edge(j,3)


    fs = []
    pieces = []

    arc = sk.Arc(C,u,v,r,phi)
    fs.append(fl.ArcField(1.0,arc,[1.0,1.0],[2.0,1.0],[1.0,1.0]))
    pieces.append((arc,[0,1,2,3]))

    A = g.nodes[0]
    B = g.nodes[i]
    n = normalize(np.cross(np.array([0.0,1.0,1.0]),B-A))
    seg = sk.Segment.make_segment(A,B,n)
    fs.append(fl.SegmentField(1.0,seg))
    pieces.append((seg,[0,i]))


    A = g.nodes[3]
    B = g.nodes[j]
    n = normalize(np.cross(np.array([1.0,1.0,0.0]),B-A))
    seg = sk.Segment.make_segment(A,B,n)
    fs.append(fl.SegmentField(1.0,seg))
    pieces.append((seg,[3,j]))


    field = fl.MultiField(fs)

    return g,field,pieces

def combined_scaff():

    C = np.array([0.0,0.0,0.0])
    r = 5.0
    phi = np.pi*0.5
    u = np.array([1.0,0.0,0.0])
    v = np.array([0.0,1.0,0.0])

    nodes = arc_to_nodes(C,u,v,r,phi)

    g = gr.Graph()
    for n in nodes:
        g.add_node(n)
    for i in range(3):
        g.add_edge(i,i+1)

    i = g.add_node(C)
    g.add_edge(i,0)

    j = g.add_node(C)
    g.add_edge(j,3)


    fs = []
    pieces = []

    arc = sk.Arc(C,u,v,r,phi)
    fs.append(fl.ArcField(1.0,arc))
    pieces.append((arc,[0,1,2,3]))


    A = g.nodes[0]
    B = g.nodes[i]
    n = np.array([0.0,0.0,1.0])
    seg = sk.Segment.make_segment(A,B,n)
    fs.append(fl.SegmentField(1.0,seg))
    pieces.append((seg,[0,i]))


    A = g.nodes[3]
    B = g.nodes[j]
    n = np.array([0.0,0.0,1.0])
    seg = sk.Segment.make_segment(A,B,n)
    fs.append(fl.SegmentField(1.0,seg))
    pieces.append((seg,[3,j]))


    field = fl.MultiField(fs)

    return g,field,pieces


def dragon():
    g = gr.Graph()
    g.read_from_graph_file("/user/afuentes/home/Work/Models/Zanni/SkelExemple/Dragon_sym.graph")


    def find_fitting(seg,ri,rf):
        A,B = seg.extremities
        n = seg.get_normal_at(0.0)
        rs = np.array([ri,rf])
        def dist(k):
            f = fl.SegmentField(seg.l/5.0,seg,b=k*rs,c=k*rs)
            QA = f.shoot_ray(A,n,0.1)
            QB = f.shoot_ray(B,n,0.1)
            d = (nla.norm(QA-A)-ri)**2 + (nla.norm(QB-B)-rf)**2
            return d

        sol = sop.minimize_scalar(dist,bounds=(0.1,100.0),method='bounded')
        k = sol.x
        print("Fitting error={} scaling factor={}".format(sol.fun,k))
        return k*rs


    fs = []
    pieces = []
    w = np.array([1.0,0.0,0.0])
    for i,j in g.edges:
        A,B = g.nodes[i],g.nodes[j]
        Ni,Nj = float(len(g.incident_edges[i])),float(len(g.incident_edges[j]))
        a = [1.0/Ni,1.0/Nj]
        a = [1.0,1.0]
        n = normalize(np.cross(B-A,w))
        seg = sk.Segment.make_segment(A,B,n)
        ri,rf = g.data["radii"][i],g.data["radii"][j]
        rs = find_fitting(seg,ri,rf)
        print("Found fitting rs={} for {}".format(rs,[ri,rf]))
        fs.append(fl.SegmentField(seg.l/5.0,seg,a=a,b=rs,c=rs))
        pieces.append((seg,[i,j]))

    field = fl.MultiField(fs)

    # ts = np.linspace(0.0,1.0,5)
    # for skel,(i,j) in pieces:
    #     bb,cc = skel.field.b,skel.field.c
    #     rs = [max(bb[0]*(1.0-t) + bb[1]*t,cc[0]*(1.0-t) + cc[1]*t) for t in ts]
    #     print rs,g.data["radii"][i],g.data["radii"][j]

    # print "------------------------"

    g.sym_file = "/user/afuentes/home/Work/Models/Zanni/SkelExemple/Dragon_sym.sym"

    return g,field,pieces


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
        arcs.append((Ce,u,v,r,phi))

    g = gr.Graph()
    fs = []
    pieces = []

    #floor part
    i = g.add_node(D)
    j = g.add_node(C)
    g.add_edge(0,1)
    seg = sk.Segment.make_segment(D,C,z,)
    fs.append(fl.SegmentField(1.0,seg,a,[3.0,3.0],[1.2,1.2]))
    pieces.append((seg,[i,j]))


    #node for the heads
    g.add_node(M) #index 2

    for arc,b,c in zip(arcs,bb,cc):
        ni = [g.add_node(n) for n in arc_to_nodes(*arc)]
        g.add_edge(ni[0],ni[1])
        g.add_edge(ni[1],ni[2])
        g.add_edge(ni[2],ni[3])
        C,u,v,r,phi = arc
        arc_sk = sk.Arc(C,u,v,r,phi)
        fs.append(fl.ArcField(1.0,arc_sk,a,b,c))
        pieces.append((arc_sk,ni))

    #baby head
    k = g.find_node_from_point(L)
    g.add_edge(2,k)
    seg = sk.Segment.make_segment(M,L,z)
    fs.append(fl.SegmentField(1.0,seg,[0.5,1.0],[1.1,1.1],[1.3,1.3]))
    pieces.append((seg,[2,k]))

    #mother head
    k = g.find_node_from_point(N)
    g.add_edge(2,k)
    seg = sk.Segment.make_segment(M,N,z)
    fs.append(fl.SegmentField(1.0,seg,[0.5,1.0],[1.1,1.1],[1.3,1.3]))
    pieces.append((seg,[2,k]))

    field = fl.MultiField(fs)

    return g,field,pieces

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
    pieces = []
    a = np.array([1.0,1.0])
    b = np.array([1.0,1.0])
    c = np.array([1.0,1.0])
    el = 0.5/float(len(aa))
    for i,(C,u,v,r,phi) in enumerate(aa):
        ni = [g.add_node(n) for n in arc_to_nodes(C,u,v,r,phi)]
        g.add_edge(ni[0],ni[1])
        g.add_edge(ni[1],ni[2])
        g.add_edge(ni[2],ni[3])
        arc = sk.Arc(C,u,v,r,phi)
        fs.append(fl.ArcField(1.0,arc,a,b-(el*i),c+(el*i)))
        pieces.append((arc,ni))
    field = fl.MultiField(fs)

    return g,field,pieces

def knot_g1():
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

    arcs = []
    for i,(C,u,v,r,phi) in enumerate(aa):
        ni = [g.add_node(n) for n in arc_to_nodes(C,u,v,r,phi)]
        g.add_edge(ni[0],ni[1])
        g.add_edge(ni[1],ni[2])
        g.add_edge(ni[2],ni[3])
        arc = sk.Arc(C,u,v,r,phi)
        arcs.append(arc)

    curve = sk.G1Curve(arcs)
    n0 = curve.get_normal_at(0.0)
    b0 = curve.get_binormal_at(0.0)
    n1 = curve.get_normal_at(curve.l)

    #correction angle to make things match in the endpoints
    phi = math.atan2(np.dot(n1,b0),np.dot(n1,n0))

    a = np.array([1.0,1.0])
    b = np.array([2.0,2.0])
    c = np.array([1.0,1.0])
    th = np.array([0.0,-phi])
    print("Phi={} ~ ---------------".format(phi,np.pi/phi))
    # th = np.array([0.0,0.0])

    field = fl.G1Field(1.0,curve,a,b,c,th)
    pieces = [(curve,[0,1,len(g.nodes)-1,0])]

    return g,field,pieces

def g1_segments():

    P = np.array([3.0,2.0,-4.0])
    v = np.array([-2.0,2.0,4.0])
    v/=nla.norm(v)
    n = np.cross(v,np.array([3.0,1.0,-9.0]))
    n/=nla.norm(n)

    ls = [4.0,3.0,5.0,2.0]
    segs = []
    for l in ls:
        segs.append(sk.Segment(P,v,l,n))
        P = P + v*l

    g = gr.Graph()
    for seg in segs:
        g.add_node(seg.extremities[0])
    g.add_node(segs[-1].extremities[1])
    for i in range(len(segs)):
        g.add_edge(i,i+1)

    curve = sk.G1Curve(segs)
    pieces = [(curve,range(len(g.nodes)))]
    pieces = [(curve,[0,1,len(g.nodes)-2,len(g.nodes)-1])]

    a = np.array([1.0,1.0])
    b = np.array([2.0,2.0])
    c = np.array([1.0,1.0])
    # th = np.array([0.0,-phi])
    th = np.array([0.0,0.0])

    field = fl.G1Field(1.0,curve,a,b,c,th)

    return g,field,pieces

def g1_segments2():

    P = np.array([0.0,0.0,0.0])
    v = np.array([1.0,0.0,0.0])
    n = np.array([0.0,0.0,1.0])

    ls = [4.0,3.0,5.0,2.0]
    segs = []
    for l in ls:
        segs.append(sk.Segment(P,v,l,n))
        P = P+v*l

    g = gr.Graph()

    for seg in segs:
        g.add_node(seg.extremities[0])
    g.add_node(segs[-1].extremities[1])

    for i in range(len(segs)):
        g.add_edge(i,i+1)

    curve = sk.G1Curve(segs)
    # pieces = [(curve,range(len(g.nodes)))]
    pieces = [(curve,[0,1,len(g.nodes)-2,len(g.nodes)-1])]

    # for t in np.linspace(0.0,1.0):
    #     print "for t= {}".format(t*curve.l),
    #     print "    P={}".format(curve.get_point_at(t*curve.l))


    field = fl.G1Field(1.0,curve)

    return g,field,pieces


def compute(g,field,pieces,min_subdivs=4,quads_num=4,split_output=False,parallel_ray_shooting=True,regular=False):

    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = min_subdivs
    if g.data["radii"]:
        scaff.set_radii(g.data["radii"])

    try:
        scaff.read_symmetries(g.sym_file)
    except:
        print( "no symmetries file")

    scaff.set_regular(regular)
    # scaff.set_regular(True)
    scaff.long_arc_angle = np.pi/2.0
    scaff.min_subdivs = min_subdivs


    s = timeit.default_timer()
    scaff.compute_scaffold()
    e = timeit.default_timer()
    print( "scaff time={}".format(e-s))

    mesher = ms.Mesher(scaff,field,pieces)

    mesher.quads_num = quads_num
    mesher.split_output = split_output
    mesher.parallel_ray_shooting = parallel_ray_shooting

    vis = scaff.get_axel_visualization()

    s = timeit.default_timer()
    mesher.compute()
    e = timeit.default_timer()
    print( "mesh time={}".format(e-s))

    mesher.draw(vis)
    vis.show()


if __name__=="__main__":
    # compute(*dragon())
    # compute(*segment_scaff(),quads_num=20,min_subdivs=16)
    # compute(*arc_scaff(),quads_num=20,min_subdivs=16)
    # compute(*combined_scaff(),quads_num=20,min_subdivs=16)
    # compute(*combined_scaff2(),quads_num=20,min_subdivs=16)
    # compute(*fertility(),quads_num=20,min_subdivs=8)
    # compute(*knot())
    # compute(*knot_g1(),quads_num=200,min_subdivs=12)
    # compute(*g1_segments2(),quads_num=20)
    compute(*g1_segments(),quads_num=20,min_subdivs=12)
