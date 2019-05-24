import PySkelton as pk

g = pk.Graph() #create a graph of the skeleton
g.add_node([0.0,0.0,0.0])
g.add_node([3.0,0.0,-4.0])
g.add_edge(0,1)

scaff = pk.Scaffolder(g)
#scaff.min_cell_quads = 8
scaff.long_arc_angle = 2*pk.pi/8
scaff.compute_scaffold()

vis = pk.get_axel_visualization("axl")
scaff.draw(vis)
vis.show()
