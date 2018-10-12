import math
import numpy as np
import scipy.optimize as sco
import timeit

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


def compute_points(f,level_set,N=20,M=40):

    u = np.array([0.0,1.0,0.0])
    v = np.array([0.0,0.0,1.0])

    mm = [u*math.cos(th) + v*math.sin(th) for th in np.linspace(0.0,2.0*math.pi,N,endpoint=False)]

    ts = np.linspace(0.0,f.skel.l,M)
    Qs = [f.skel.get_point_at(t) for t in ts]
    FsT = [f.skel.get_frame_at(t).T for t in ts]

    ps = []
    for m in mm:
        ps.extend(f.shoot_ray(Q,(m*FT).A1,level_set) for Q,FT in zip(Qs,FsT))

    return ps

def show(f,ps):
    vis = visual.VisualizationAxel()

    vis.add_points(ps,name="cloud",color="magenta")

    # M = 40
    # ts = np.linspace(0.0,f.skel.l,M)
    # FsT = [f.skel.get_frame_at(t).T.A for t in ts]
    # Qs = [f.skel.get_point_at(t) for t in ts]

    # for F,Q in zip(FsT,Qs):
    #     vis.add_polyline([Q,Q+F[0]],name="T",color="red")
    #     vis.add_polyline([Q,Q+F[1]],name="N",color="green")
    #     vis.add_polyline([Q,Q+F[2]],name="B",color="blue")

    vis.show()

def do_fitting(skel,ps,level_set,R,o,vals):


    f = fl.make_field(R,skel,a=[1.0,1.0],b=[3.0,3.0],c=[3.0,3.0],th=[0.0,0.0])
    x0 = np.array(list(f.a) + list(f.b) + list(f.c) + list(f.th))

    def update_f(params):
        if (0 in vals) and (1 in vals):
            a0 = params[vals.index(0)]
            a1 = params[vals.index(1)]
            f.a = np.array([a0,a1])
        if (2 in vals) and (3 in vals):
            b0 = params[vals.index(2)]
            b1 = params[vals.index(3)]
            f.b = np.array([b0,b1])
        if (4 in vals) and (5 in vals):
            c0 = params[vals.index(4)]
            c1 = params[vals.index(5)]
            f.c = np.array([c0,c1])
        if (6 in vals) and (7 in vals):
            th0 = params[vals.index(6)]
            th1 = params[vals.index(7)]
            f.th = np.array([th0,th1])

    def fun(params):
        update_f(params)
        return np.array([level_set-f.eval(p) for p in ps])

    def fun2(xdata,*params):
        update_f(params)
        return map(f.eval,xdata)

    def jac(params):
        update_f(params)
        return np.array([f.parametric_gradient_eval(p,vals=vals) for p in ps])


    x0 = np.array(x0[vals])
    low = np.zeros(len(x0))
    upp = 2.0*x0
    upp[vals.index(6)] = np.pi
    upp[vals.index(7)] = np.pi
    print low,upp

    # stime = timeit.default_timer()
    # sol = sco.least_squares(fun,x0,bounds=(low,upp))
    # etime = timeit.default_timer()
    # print "Time with least_squares={}".format(etime-stime)
    # print sol
    # print "x={}".format(sol.x)
    # print "o={}".format(o)
    # print "error={}".format(nla.norm(o-sol.x))

    stime = timeit.default_timer()
    sol = sco.curve_fit(fun2,ps,np.repeat(level_set,len(ps)),p0=x0,bounds=(low,upp))
    etime = timeit.default_timer()
    print "Time with curve_fit={}".format(etime-stime)
    print sol
    print "x={}".format(sol[0])
    print "o={}".format(o)
    print "error={}".format(nla.norm(o-sol[0]))

    # total_iters = 20
    # iter = total_iters
    # I = np.identity(len(x))
    # scale = 0.1*float(len(r(x)))
    # while iter:
    #     iter-=1
    #     y = r(x)
    #     S = sum(y_*y_ for y_ in y)
    #     if S<0.0001*len(y):
    #         print "Tolerance reached!"
    #         break

    #     J = np.matrix(Jr(x))

    #     Jt = J.T
    #     K = Jt*J
    #     # dx = np.dot((K).I*Jt,y).A1 #Gauss-Newton
    #     # dx = np.dot((K + scale*I).I*Jt,y).A1 #Levenberg
    #     dx = np.dot((K + scale*np.diag(K)).I*Jt,y).A1 #Levenberg-Marquardt
    #     print "[{}] x={} S={} dx={}".format(total_iters-iter,x,S,dx)


    #     x = x + dx




def fitting():
    a = [1.0,1.0]
    b = [2.0,2.0]
    c = [1.0,1.0]
    th = [0.0,np.pi/2.0]

    R = 1.0

    # P = np.array([0.0,0.0,0.0])
    # v = np.array([1.0,0.0,0.0])
    # l = 5.0
    # n = np.array([0.0,0.0,1.0])

    # skel = sk.Segment(P,v,l,n)


    C = np.array([0.0,0.0,0.0])
    u = np.array([1.0,0.0,0.0])
    v = np.array([0.0,1.0,0.0])
    r = 5.0
    phi = np.pi/2.0

    skel = sk.Arc(C,u,v,r,phi)

    f = fl.make_field(R,skel,a=a,b=b,c=c,th=th)

    ps = compute_points(f,level_set=0.1)

    do_fitting(skel,ps,level_set=0.1,R=1.0,o=np.array(a+b+c+th),vals=[0,1,2,3,4,5,6,7])


    show(f,ps)


if __name__=="__main__":
    fitting()
