import PySkelton as pk


C = [0.0,0.0,0.0]
u = [1.0,0.0,0.0]
v = [0.0,1.0,0.0]
r = 5.0
phi = pk.pi/2
arc = pk.Arc(C,u,v,r,phi)

g = pk.Graph()
idxs = [g.add_node(x) for x in pk.arc_to_nodes(C,u,v,r,phi)]
for i in range(len(idxs)-1): g.add_edge(i,i+1)
R = 1.0
field = pk.make_field(R,arc,b=[0.26,1.05],c=[0.26,0.26],th=[0.0,2*pk.pi])

scaff = pk.Scaffolder(g)
scaff.min_cell_quads = 40
scaff.compute_scaffold()
pieces = [(arc,idxs)]
mesher = pk.Mesher(scaff,field,pieces)
mesher.max_quads_size = 0.2
mesher.cap_quads_num = 10
mesher.compute()

mesher.draw(pk.get_axel_visualization()).show()
