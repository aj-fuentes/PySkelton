import PySkelton as pk
import numpy as np
import math

#Knot curve
def knot(t):
    return [-10*np.cos(t)-2*np.cos(5*t)+15*math.sin(2*t),
        -15*np.cos(2*t)+10*math.sin(t)-2*math.sin(5*t),
        10*np.cos(3*t)]
#Knot tangent
def knot_tang(t):
    return pk.normalize([
        10*math.sin(t) +10*math.sin(5*t)+ 30*np.cos(2*t),
        30*math.sin(2*t)+10*np.cos(t)-10*np.cos(5*t),
        -30*math.sin(3*t)])

#compute biarcs approximation
bas = pk.biarcs_from_curve(
    knot,knot_tang,0.0,2*pk.pi,20)
arcs = [pk.Arc(*data) for data in bas]

g = pk.Graph()
for data in bas:
    idxs = [g.add_node(n) for n in pk.arc_to_nodes(*data)]
    g.add_edge(idxs[0],idxs[1])
    g.add_edge(idxs[1],idxs[2])
    g.add_edge(idxs[2],idxs[3])

curve = pk.G1Curve(arcs)

n0 = curve.get_normal_at(0.0)
b0 = curve.get_binormal_at(0.0)
n1 = curve.get_normal_at(curve.l)

#correction angle to make things match in the endpoints
phi = math.atan2(np.dot(n1,b0),np.dot(n1,n0))

a = np.array([1.0,1.0])
b = np.array([2.0,2.0])
c = np.array([1.0,1.0])
th = np.array([0.0,6*np.pi-phi])
th = np.array([0.0,-phi])
# th = np.array([0.0,0.0])
print("Phi={} ~ ---------------".format(phi,np.pi/phi))
# th = np.array([0.0,0.0])

field = pk.G1Field(1.0,curve,a,b,c,th)
pieces = [(curve,[0,1,len(g.nodes)-1,0])]

scaff = pk.Scaffolder(g)
scaff.min_cell_quads = 12
scaff.compute_scaffold()
mesher = pk.Mesher(scaff,field,pieces)
# mesher.max_quads_size = 0.2
mesher.quads_num = 200
mesher.cap_quads_num = 10
mesher.compute()

mesher.draw(pk.get_axel_visualization()).show()
