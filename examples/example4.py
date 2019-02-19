import PySkelton as pk

g = pk.Graph()
g.read_from_graph_file("Dragon.graph")
scaff = pk.Scaffolder(g)
scaff.read_symmetries("Dragon.sym")
scaff.long_arc_angle = pk.pi/10
scaff.set_radii(g.data["radii"])
scaff.compute_scaffold()
scaff.get_axel_visualization().show()
