import math
import numpy as np

from _math import *
import scaffolder as sc
import biarcs
import graph as gr
import mesher as ms
import skeleton as sk
import field as fl
import visualization as visual


import scipy.optimize as sop

import matplotlib.pyplot as plt
import matplotlib.widgets as wd

def rad_vals():
    P = np.array([1.0,4.0,0.24134])
    v = normalize(np.array([123.,12.1,23.9]))
    n = normalize(np.cross(np.array([1.0,0.0,0.0]),v))
    w = normalize(np.cross(v,n))


    l = 2.0
    max_R = 2.0
    min_R = 0.01

    N = 50
    ts = np.linspace(0.0,1.0,N)

    seg = sk.Segment(P,v,l,n)
    Ps = [seg.get_point_at(t*seg.l) for t in ts]
    def compute_data(l=l,R=1.0,alpha=1.0,level_set=0.1):
        f = fl.SegmentField(R,seg,a=[1.0/math.sqrt(alpha),1.0/math.sqrt(alpha)])
        Qs = [f.shoot_ray(P0,n,0.1) for P0,t in zip(Ps,ts)]
        return np.array([nla.norm(Q-P0) for Q,P0 in zip(Qs,Ps)])


    plt.subplots_adjust(bottom=0.2)
    ys = compute_data()

    kw,  = plt.plot(ts*l,ys)


    plt.axis([0,l,0.0,2.0])
    plt.grid()


    rad_sl = wd.Slider(plt.axes([0.25, 0.1, 0.65, 0.03]),"R",min_R,max_R,valinit=1.0, valstep=0.01)
    def update(val):
        ys = compute_data(l=l,R=val)
        kw.set_ydata(ys)
        plt.gcf().canvas.draw_idle()
    rad_sl.on_changed(update)

    plt.show()

def compute_dist():
    P = np.array([1.0,4.0,0.24134])
    v = normalize(np.array([123.,12.1,23.9]))
    n = normalize(np.cross(np.array([1.0,0.0,0.0]),v))
    w = normalize(np.cross(v,n))

    l = 12.0
    R = l/5.0
    alpha = 1.0
    level_set = 0.1

    seg = sk.Segment(P,v,l,n)
    f = fl.SegmentField(R,seg,a=[1.0/math.sqrt(alpha),1.0/math.sqrt(alpha)])

    x = f.eval(P)
    r = norm(f.shoot_ray(P,n,level_set)-P)

    print "f(P)={} r={}".format(x,r)

    N=20
    vs = [v*math.cos(t)+w*math.sin(t) for t in np.linspace(0.0,2*np.pi,N,endpoint=False)]

    fs = []
    for v in vs:
        seg = sk.Segment(P,v,5.0,n)
        alpha_ = N*N*alpha
        f = fl.SegmentField(R,seg,a=[1.0/math.sqrt(alpha_),1.0/math.sqrt(alpha_)])
        fs.append(f)

    f = fl.MultiField(fs)

    x = f.eval(P)
    r = norm(f.shoot_ray(P,n,level_set)-P)

    print "f(P)={} r={} multi".format(x,r)

def kernel_vals():
    P = np.array([1.0,4.0,0.24134])
    v = normalize(np.array([123.,12.1,23.9]))
    n = normalize(np.cross(np.array([1.0,0.0,0.0]),v))
    w = normalize(np.cross(v,n))


    max_l = 10.0
    min_l = 1.0

    N = 50
    ts = np.linspace(0.0,1.0,N)

    def compute_data(l=5.0,R=1.0,alpha=1.0,level_set=0.1):
        seg = sk.Segment(P,v,l,n)
        f = fl.SegmentField(R,seg,a=[1.0/math.sqrt(alpha),1.0/math.sqrt(alpha)])
        return np.array([f.eval(seg.get_point_at(t*seg.l)) for t in ts])


    plt.subplots_adjust(bottom=0.2)
    ys = compute_data(l=5.0,R=1.0)
    ys2 = compute_data(l=5.0,R=1.0)

    kw,  = plt.plot(ts*5.0,ys)
    kw2, = plt.plot(ts*5.0,ys2)

    ts_segs = ts[ys>1.9999][[0,-1]]*5.0
    vls  = plt.vlines(ts_segs,0.0,2.1,linestyles='dashed',color=kw.get_color())
    ts_segs2 = ts[ys2>1.9999][[0,-1]]*5.0
    vls2 = plt.vlines(ts_segs2,0.0,2.1,linestyles='dashed',color=kw2.get_color())

    ans = [plt.text(t,0.95,"{:.1f}".format(t)) for t in ts_segs]
    ans2 = [plt.text(t,0.95,"{:.1f}".format(t)) for t in ts_segs2]

    plt.axis([0,max_l,0.9,2.1])
    plt.grid()


    len_sl = wd.Slider(plt.axes([0.25, 0.1, 0.65, 0.03]),"Len",min_l,max_l,valinit=5.0, valstep=0.1)
    def update(val):
        kw.set_xdata(ts*val)
        kw2.set_xdata(ts*val)

        ys = compute_data(l=val,R=val/5.0)
        kw.set_ydata(ys)
        ys2 = compute_data(l=val,R=1.0)
        kw2.set_ydata(ys2)


        def update_lines(ls,ans,ts_segs,val):
            if len(ts_segs):
                ls.set_alpha(1.0)
                ts_ = ts_segs[[0,-1]]*val
                segs = [ [(t,0.0),(t,2.1)]  for t in ts_]
                ls.set_segments(segs)
                for ann,t in zip(ans,ts_):
                    ann.set_alpha(1.0)
                    ann.set_x(t)
                    ann.set_text("{:.1f}".format(t))
            else:
                ls.set_alpha(0.0)
                for ann in ans:
                    ann.set_alpha(0.0)

        update_lines(vls,ans,ts[ys>1.9999],val)
        update_lines(vls2,ans2,ts[ys2>1.9999],val)


        # plt.vlines(,)

        plt.gcf().canvas.draw_idle()
    len_sl.on_changed(update)


    plt.show()


def test_rad():
    P = np.array(10*np.random.rand(3))
    v = normalize(np.random.rand(3))
    n = normalize(np.cross(np.array([1.0,0.0,0.0]),v))
    w = normalize(np.cross(v,n))


    r = 2.0
    l = 5.0
    level_set = 0.1
    N = 1.0

    alpha = N*N
    a = np.repeat(1.0/math.sqrt(alpha),2)

    R = 1.0
    R = 1.0/math.sqrt(1.0-math.pow(0.5*level_set/(alpha*alpha*alpha),2.0/7.0)*alpha)


    R_ = R*math.pow(0.5*level_set/(alpha*alpha*alpha),1.0/7.0)
    beta = (R*R-alpha*(R_*R_))/(r*r)
    b = np.repeat(1.0/math.sqrt(beta),2)

    seg = sk.Segment(P,v,l,n)
    f = fl.SegmentField(R,seg,a=a,b=b)

    print "l={} level_set={}".format(l,level_set)
    print "R={} R_={}".format(R,R_)
    print "alpha={} beta={}".format(alpha,beta)
    print "b={} a={}".format(b[0],a[0])
    print "r={}".format(r)
    print "f(P)={} (~ 1/{:1.0f}={})".format(f.eval(P),N,1.0/N)
    print " f(P+R_T+rN)={}".format(f.eval(P+R_*v+r*n))
    print "f(P+l/2T+rN)={}".format(f.eval(P+(l/2.0)*v+r*n))

def test_shape():
    P = np.array(10*np.random.rand(3))
    v = normalize(np.random.rand(3))
    n = normalize(np.cross(np.array([1.0,0.0,0.0]),v))
    w = normalize(np.cross(v,n))

    l = 10.0
    level_set = 0.1
    R = 0.5


    def get_abR_(R,r,level_set,N=1.0):
        alpha = N*N
        a = np.repeat(1.0/math.sqrt(alpha),2)

        R_ = R*math.pow(0.5*level_set/(alpha*alpha*alpha),1.0/7.0)
        beta = (R*R-alpha*(R_*R_))/(r*r)
        b = np.repeat(1.0/math.sqrt(beta),2)

        return a,b,R_



    g = gr.Graph()
    g.add_node(P)

    a0,b0,R0 = get_abR_(R,2.0,level_set)
    g.add_node(P+2*R0*v)

    a1,b1,R1 = get_abR_(R,4.0,level_set)
    g.add_node(P+(l-2*R1)*v)

    g.add_node(P+l*v)

    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(2,3)

    pieces = []
    fs = []


    seg = sk.Segment(g.nodes[0],v,2*R0,n)
    print seg.get_point_at(2*R0)-g.nodes[1]
    f = fl.SegmentField(R,seg,a=a0,b=b0,c=b0)
    pieces.append((seg,[0,1]))
    fs.append(f)

    seg = sk.Segment(g.nodes[2],v,2*R1,n)
    print seg.get_point_at(2*R1)-g.nodes[3]
    f = fl.SegmentField(R,seg,a=a1,b=b1,c=b1)
    pieces.append((seg,[2,3]))
    fs.append(f)

    seg = sk.Segment(g.nodes[1],v,l-2*R0-2*R1,n)
    print seg.get_point_at(0.0)-g.nodes[1]
    print seg.get_point_at(l-2*R0-2*R1)-g.nodes[2]
    f = fl.SegmentField(R,seg,a=[a0[0],a1[0]],b=[b0[0],b1[1]],c=[b0[0],b1[1]])
    pieces.append((seg,[1,2]))
    fs.append(f)

    f = fl.MultiField(fs)

    from _test import compute
    compute(g,f,pieces,min_subdivs=20,quads_num=16)



def dragon_sym():
    g = gr.Graph()
    g.read_from_graph_file("/user/afuentes/home/Work/Models/Zanni/SkelExemple/Dragon.graph")

    sym = []

    def fix(g,i,j):
        g.nodes[j][1:]=g.nodes[i][1:]
        g.nodes[j][0]=-g.nodes[i][0]
    def zero(g,i):
        g.nodes[i][0]=0.0

    fix(g,24,70)
    fix(g,32,62)
    fix(g,35,59)
    fix(g,38,56)

    for i in range(41,54):
        zero(g,i)

    for i,A in enumerate(g.nodes):
        found = False
        for j,B in enumerate(g.nodes):
            if np.all(np.abs(A[1:]-B[1:])<0.001) and i!=j:
                print "{} {} {} {} {}".format(found,i,j,A,B)
                sym.append(j)
                found = True
        if not found:
            print "{} {} {} {} {}".format(found,i,"-",A,"[-------]")
            sym.append(i)

    print len(sym)
    print sym

    g.save_to_graph_file("/user/afuentes/home/Work/Models/Zanni/SkelExemple/Dragon_sym.graph")

    #24-70
    #59-35
    #32-62
    #38-56

    ##41----53 =0


def pgradient():
    P = np.array(10*np.random.rand(3))
    v = normalize(np.random.rand(3))
    n = normalize(np.cross(np.array([1.0,0.0,0.0]),v))
    w = normalize(np.cross(v,n))


    l = 10.0
    level_set = 0.1
    R = 0.5
    N = 1.0

    def get_abcR_(R,rb,rc,level_set,N=1.0):
        alpha = N*N
        a = np.repeat(1.0/math.sqrt(alpha),2)

        R_ = R*math.pow(0.5*level_set/(alpha*alpha*alpha),1.0/7.0)
        beta = (R*R-alpha*(R_*R_))/(rb*rb)
        gamma = (R*R-alpha*(R_*R_))/(rc*rc)
        b = np.repeat(1.0/math.sqrt(beta),2)
        c = np.repeat(1.0/math.sqrt(gamma),2)

        return a,b,c,R_

    rb = 2.0
    rc = 1.0
    a,b,c,R_ = get_abcR_(R,rb,rc,level_set,N)

    seg = sk.Segment(P,v,l,n)
    f = fl.SegmentField(R,seg,a=a,b=b,c=c)

    print "l={} level_set={}".format(l,level_set)
    print "R={} R_={}".format(R,R_)
    print "alpha={} beta={} gamma={}".format(1.0/a[0]/a[0],1.0/b[0]/b[0],1.0/c[0]/c[0])
    print "b={} a={}".format(b[0],a[0])
    print "rb={} rc={}".format(rb,rc)
    print "f(P)={} (~ 1/{:1.0f}={})".format(f.eval(P),N,1.0/N)
    print " f(P+R_T+rbN)={}".format(f.eval(P+R_*v+rb*n))
    print " f(P+R_T+rbB)={}".format(f.eval(P+R_*v+rb*w))
    print " f(P+R_T+rcN)={}".format(f.eval(P+R_*v+rc*n))
    print " f(P+R_T+rcB)={}".format(f.eval(P+R_*v+rc*w))
    print "f(P+l/2T+rbN)={}".format(f.eval(P+(l/2.0)*v+rb*n))
    print "f(P+l/2T+rcB)={}".format(f.eval(P+(l/2.0)*v+rc*w))
    print "gradient         f(P)={}".format(f.parametric_gradient_eval(P))
    print "gradient f(P+2.0*T+0.5N)={}".format(f.parametric_gradient_eval(P+2.0*v+0.5*n))

def test_tang_dist():
    P = np.array(10*np.random.rand(3))
    v = normalize(np.random.rand(3))
    n = normalize(np.cross(np.array([1.0,0.0,0.0]),v))
    w = normalize(np.cross(v,n))

    l = 10.0
    level_set = 0.1
    R = 0.5
    a = [1.0,1.0]
    b = [4.0,4.0]
    c = [1.0,1.0]

    seg = sk.Segment(P,v,l,n)
    f = fl.SegmentField(R,seg,a=a,b=b,c=c)

    Q = f.shoot_ray(P,-v,level_set)
    print "tangential radius {}".format(nla.norm(Q-P))
    print "ratio with radius {}".format(nla.norm(Q-P)/R)


if __name__=="__main__":
    # compute_dist()
    # kernel_vals()
    # rad_vals()
    # test_rad()
    # test_shape()
    # dragon_sym()
    # pgradient()
    test_tang_dist()
