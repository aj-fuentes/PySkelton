import numpy as np
import scipy.optimize as sop
import timeit

import skeleton as sk
import field as fl


def do_fitting(skel,ps,level_set,R,low,upp,vals):

    ps = map(np.array,ps)
    level_set = float(level_set)
    R = float(R)
    low = np.array(low)
    upp = np.array(upp)

    f = fl.make_field(R,skel)
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

    def fun2(xdata,*params):
        update_f(params)
        return map(f.eval,xdata)


    p0 = upp - 0.1
    update_f(p0)

    stime = timeit.default_timer()
    sol = sop.curve_fit(fun2,ps,np.repeat(level_set,len(ps)),p0=p0,bounds=(low,upp))
    etime = timeit.default_timer()
    print "Time with curve_fit={}".format(etime-stime)

    return sol[0],f

