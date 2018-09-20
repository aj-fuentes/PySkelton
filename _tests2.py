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



def compute_dist():
    P = np.array([1.0,4.0,0.24134])
    v = normalize(np.array([123.,12.1,23.9]))
    n = normalize(np.cross(np.array([1.0,0.0,0.0]),v))
    w = normalize(np.cross(v,n))

    R = 1.0
    alpha = 1.0
    level_set = 0.1

    seg = sk.Segment(P,v,5.0,n)
    f = fl.SegmentField(R,seg,a=[1.0/math.sqrt(alpha),1.0/math.sqrt(alpha)])

    x = f.eval(P)
    y = 16.0/35.0*R/math.sqrt(alpha)
    r = norm(f.shoot_ray(P,n,level_set)-P)

    print "f(P)={} 16/35*R/sqrt(alpha)={}".format(x,y)
    print "r={}".format(r)

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
    y = 16.0/35.0*R/math.sqrt(alpha)
    r = norm(f.shoot_ray(P,n,level_set)-P)

    print "f(P)={} 16/35*R/sqrt(alpha)={}".format(x,y)
    print "r={}".format(r)


if __name__=="__main__":
    compute_dist()
