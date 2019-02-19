import PySkelton as pk

g = pk.Graph()
g.read_from_graph_file("simple.graph")
scaff = pk.Scaffolder(g)
scaff.set_radii(g.data["radii"])
scaff.compute_scaffold()
scaff.get_axel_visualization().show()
