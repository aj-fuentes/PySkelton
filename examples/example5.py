import PySkelton as pk

g = pk.Graph() #create a graph of the skeleton
g.add_node([0.0,0.0,0.0])
g.add_node([3.0,0.0,-4.0])
g.add_edge(0,1)

n = [0.0,1.0,0.0]
A,B = g.nodes[0],g.nodes[1]
seg = pk.Segment.make_segment(A,B,n)
pieces = [(seg,[0,1])]

R = 1.0
# field = pk.make_field(R,seg)
# field = pk.make_field(R,seg,b=[0.26,1.05],c=[0.26,1.05])
# field = pk.make_field(R,seg,b=[0.26,0.26],c=[1.05,1.05])
# field = pk.make_field(R,seg,c=[0.26,0.26],b=[1.05,1.05])
# field = pk.make_field(R,seg,c=[0.26,0.26],b=[1.05,1.05],th=[0.0,4*pk.pi])
# field = pk.make_field(R,seg,c=[0.26,0.26],b=[1.05,1.05],th=[0.0,2*pk.pi])
field = pk.make_field(R,seg,c=[0.26,0.26],b=[0.26,1.05],th=[0.0,2*pk.pi])

scaff = pk.Scaffolder(g)
scaff.min_cell_quads = 40
scaff.compute_scaffold()
mesher = pk.Mesher(scaff,field,pieces)
mesher.max_quads_size = 0.2
mesher.cap_quads_num = 10
mesher.compute()

vis = pk.get_axel_visualization()
mesher.draw(vis)
vis.show()
