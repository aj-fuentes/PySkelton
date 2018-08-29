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
    nodes = arc_to_nodes(C,u,v,r,phi)
    arc = nodes_to_arc(*nodes)

    g = gr.Graph()
    for n in nodes:
        g.add_node(n)
    for i in range(3):
        g.add_edge(i,i+1)

    g.add_arc([0,1,2,3])

    scaff = sc.Scaffolder(g)
    scaff.min_subdivs = 16

    scaff.compute_scaffold()

    arc_skeleton = sk.Arc(C,u,v,r,phi)
    field = fl.ArcField(1.0,arc_skeleton,[1.0,1.0],[2.0,1.0],[1.0,1.0])

    mesher = ms.Mesher(scaff,field)
    # mesher.quads_num=1

    vis = scaff.get_axel_visualization()

    vis.add_polyline(arc_points(arc),name="arc",color="black")

    mesher.draw(vis)

    vis.show()


if __name__=="__main__":
    arc_scaff()
