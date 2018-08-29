import subprocess

from ConfigParser import RawConfigParser

import numpy as np
import time

#colors
blue = [0,0,255]
cyan = [0,255,255]
green = [0,255,0]
magenta=[255,0,255]
dark_yellow=[100,100,0]
yellow=[255,255,0]
dark_green=[0,190,0]
red = [255,0,0]
black = [0,0,0]
gray = [100,100,100]

default_palette = [red,yellow,green,cyan,blue,magenta]
pastel_palette = [[(x+200)/2 for x in color] for color in default_palette]
dark_palette = [[(x)/2 for x in color] for color in default_palette]

vtk = None
try:
    import vtk
except:
    print "No vtk in your system"

def get_axel_visualization():
    return VisualizationAxel()

def _verify_color(color):
    if isinstance(color,str):
        return globals()[color]
    else:
        return color

class Visualization(object):

    def __init__(self):

        self.subdivision_number = 0

        self.meshes = {} # name -> points,facets,color
        self.normals = {} #name -> normals
        self.polylines = {} # name -> points,lens,color
        self.points = {} # name -> points,color

    def set_subdivision_number(self,n):
        self.subdivision_number = n

    def add_mesh(self, mesh_points, mesh_facets, color=[0,0,255], name=None):
        color = _verify_color(color)
        if name is None:
            name = "Mesh%d" % len(self.meshes)
        if name in self.meshes.keys():
            n = len(self.meshes[name][0])
            self.meshes[name][0].extend(mesh_points)
            fs = list(map(list,mesh_facets))
            for f in fs:
                for i in xrange(len(f)):
                    f[i] += n
            self.meshes[name][1].extend(fs)
        else:
            self.meshes[name] = (list(mesh_points), list(mesh_facets), list(color) )

    def add_normals(self, normals, name):
        if self.normals.has_key(name):
            self.normals[name].extend(normals)
        else:
            self.normals[name] = list(normals)

    def add_polyline(self, poly_points, color=[255,0,0], name=None):
        color = _verify_color(color)
        if name is None:
            name = "Polyline%d" % len(self.polylines)
        if name in self.polylines.keys():
            self.polylines[name][0].extend(list(poly_points))
            self.polylines[name][1].append(len(poly_points))
        else:
            self.polylines[name] = (list(poly_points), [len(poly_points)], list(color) )

    def add_points(self, points, color=[0,255,0], name=None):
        color = _verify_color(color)
        if name is None:
            name = "Points%d" % len(self.points)

        if name in self.points.keys():
            self.points[name][0].extend(list(points))
        else:
            self.points[name] = (list(points), list(color) )

    def show(self):
        raise NotImplementedError()

class VisualizationVTK(Visualization):

    def __init__(self, bg_color=[255,255,255]):
        super(VisualizationVTK, self).__init__()

        self.vtk_objects = []
        self.actors = []
        self.back_color = bg_color[0]/255.0,bg_color[1]/255.0,bg_color[2]/255.0

    """Adds a mesh to the visualization
    mesh_points is a list of points supporting the mesh
    mesh_triangles is a list of tuples of 3 indices indicating the triangles of the mesh
    """
    def add_vtk_mesh(self, mesh_points, mesh_elems, opacity=1.0, pos=[0,0,0], color=magenta):

        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)

        points = vtk.vtkPoints()
        for x,y,z in mesh_points:
            points.InsertNextPoint(x,y,z)

        cells = vtk.vtkCellArray()

        for elem in mesh_elems:
            vtk_elem = vtk.vtkTriangle()
            if len(elem)==4:
                vtk_elem = vtk.vtkQuad()

            vtk_elem.GetPointIds().SetId(0, elem[0])
            vtk_elem.GetPointIds().SetId(1, elem[1])
            vtk_elem.GetPointIds().SetId(2, elem[2])
            if len(elem)==4:
                vtk_elem.GetPointIds().SetId(3, elem[3])

            colors.InsertNextTupleValue(color)
            cells.InsertNextCell(vtk_elem)

        mesh = vtk.vtkPolyData()
        mesh.SetPoints(points)
        mesh.GetCellData().SetScalars(colors)
        mesh.SetPolys(cells)

        to_triangles = vtk.vtkTriangleFilter()
        to_triangles.SetInputData(mesh)

        #subdiv = vtk.vtkLinearSubdivisionFilter()
        #subdiv = vtk.vtkLoopSubdivisionFilter()
        subdiv = vtk.vtkButterflySubdivisionFilter()
        subdiv.SetNumberOfSubdivisions(self.subdivision_number)
        subdiv.SetInputConnection(to_triangles.GetOutputPort())

        mapper = vtk.vtkDataSetMapper()
        #self.vtk_objects.append(mapper)
        if self.subdivision_number:
            mapper.SetInputConnection(subdiv.GetOutputPort())
        else:
            mapper.SetInputData(mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.AddPosition(*pos)
        actor.GetProperty().SetOpacity(opacity)
        self.actors.append(actor)

    """Adds a point cloud to the visualization
    ps is a list of points (3D)
    """
    def add_vtk_points(self, ps, color=green, opacity=1.0, point_size=4.0, pos=[0,0,0]):

        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        vertexs = vtk.vtkPolyVertex()
        vertexs.GetPointIds().SetNumberOfIds(len(ps))
        for x,y,z in ps:
            p_id = points.InsertNextPoint(x,y,z)

            vertexs.GetPointIds().SetId(p_id, p_id)

        cells.InsertNextCell(vertexs)

        cloud = vtk.vtkPolyData()
        cloud.SetPoints(points)
        cloud.SetVerts(cells)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(0, 1)
        mapper.SetScalarVisibility(1)
        mapper.SetInputData(cloud)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.AddPosition(*pos)
        actor.GetProperty().SetPointSize(point_size)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        #actor.GetProperty().SetRenderPointsAsSpheres(1)
        self.actors.append(actor)

    """Adds a polyline
    ps is a list of points (3D)
    """
    def add_vtk_polyline(self, ps, color=cyan, opacity=1.0, line_width=1.0, point_size=1.0, pos=[0,0,0]):
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        pline = vtk.vtkPolyLine()
        pline.GetPointIds().SetNumberOfIds(len(ps))
        for x,y,z in ps:
            p_id = points.InsertNextPoint(x,y,z)

            pline.GetPointIds().SetId(p_id, p_id)

        cells.InsertNextCell(pline)

        poly_line = vtk.vtkPolyData()
        poly_line.SetPoints(points)
        poly_line.SetLines(cells)

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(poly_line)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.AddPosition(*pos)
        actor.GetProperty().SetPointSize(point_size)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLineWidth(line_width)
        #actor.GetProperty().RenderPointsAsSpheresOn()
        #actor.GetProperty().RenderLinesAsTubesOn()
        self.actors.append(actor)

    # def add_delaunay(self, ps, color=[1,0,0], opacity=1.0, pos=[0,0,0]):
    #     points = vtk.vtkPoints()
    #     for i,p in enumerate(ps):
    #         points.InsertPoint(i, *p)

    #     profile = vtk.vtkPolyData()
    #     profile.SetPoints(points)

    #     # Delaunay3D is used to triangulate the points. The Tolerance is the
    #     # distance that nearly coincident points are merged
    #     # together. (Delaunay does better if points are well spaced.) The
    #     # alpha value is the radius of circumcircles, circumspheres. Any mesh
    #     # entity whose circumcircle is smaller than this value is output.
    #     delny = vtk.vtkDelaunay3D()
    #     delny.SetInputData(profile)
    #     delny.SetTolerance(0.01)
    #     delny.SetAlpha(10)
    #     #delny.SetAlphaTets(0)
    #     #delny.SetAlphaTris(1)
    #     delny.SetAlphaLines(1)
    #     delny.BoundingTriangulationOff()

    #     mapper = vtk.vtkDataSetMapper()
    #     mapper.SetInputConnection(delny.GetOutputPort())

    #     actor = vtk.vtkActor()
    #     actor.SetMapper(mapper)
    #     actor.AddPosition(*pos)
    #     actor.GetProperty().SetColor(color)
    #     actor.GetProperty().SetOpacity(opacity)

    #     self.actors.append(actor)


    def show(self):


        for mesh in self.meshes.values():
            ps, fs, color = mesh
            self.add_vtk_mesh(ps,fs,color=color)

        for poly in self.polylines.values():
            ps, ls, color = poly
            n = 0
            for l in ls:
                self.add_vtk_polyline(ps[n:n+l],color=color)
                n+=l

        for ps,color in self.points.values():
            self.add_vtk_points(ps,color=color)


        # Visualize
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        # Add actors and render
        for actor in self.actors:
            renderer.AddActor(actor)

        renderer.SetBackground(*self.back_color)
        renderWindow.SetSize(1200, 800)
        renderWindow.Render()
        renderWindowInteractor.Start()

class VisualizationAxel(Visualization):
    """docstring for Visualization2"""
    def __init__(self, axel_path="env LD_LIBRARY_PATH=/home/afuentes/Programs/Axel/lib64:/home/afuentes/Programs/Axel/lib:/user/afuentes/home/Work/Building/qt5.9-install/5.9.2/gcc_64/lib/ /home/afuentes/Programs/Axel/bin/axel",output_file="vis.axl"):
        super(VisualizationAxel, self).__init__()

        self.output_file = output_file

        self.axel_path = axel_path
        if axel_path is None:
            local = RawConfigParser()
            local.read('local.conf')
            self.axel_path = local.get('Local Paths','axel_executable_path')
            self.subdiv_path = local.get('Local Paths','subdiv_executable_path')

            self.mesh_to_subdivide = None

    def set_mesh_to_subdivide(self,name):
        self.mesh_to_subdivide = name

    def write_mesh(self, f, name):
        ps, fs, color = self.meshes[name]
        ps,inv = np.unique(ps,axis=0,return_inverse=True)

        f.write('<mesh color="%d %d %d 1" shader="" name="%s" size="0.05">\n' % tuple(color + [name]) )
        f.write('\t<count>%d 0 %d</count>\n' % (len(ps),len(fs)) ) #points edges facets
        f.write('\t<points>\n')
        for p in ps:
            f.write('\t\t%f %f %f\n' % tuple(p))
        f.write('\t</points>\n')

        if self.normals.has_key(name):
            ns = self.normals[name]
            if len(ns) != len(ps):
                print "Error: not the same number of normals and points in the mesh: %" % name
            else:
                f.write('\t<normals>\n')
                for n in ns:
                    f.write('\t\t%f %f %f\n' % tuple(n))
                f.write('\t</normals>\n')

        #f.write('\t<edges></edges>\n')
        f.write('\t<faces>\n')
        for fa_ in fs:
            fa = [inv[x] for x in fa_]
            f.write('\t\t%d %s\n' % (len(fa),' '.join(map(str,fa))))
        f.write('\t</faces>\n')
        f.write('</mesh>\n')

    def write_polyline(self, f, name):
        ps, ls, color = self.polylines[name]
        ps,inv = np.unique(ps,axis=0,return_inverse=True)

        f.write('<mesh color="%d %d %d 1" shader="" name="%s" size="0.2">\n' % tuple(color + [name]) )
        f.write('\t<count>%d %d 0</count>\n' % (len(ps),len(ls)) ) #points edges facets
        f.write('\t<points>\n')
        for p in ps:
            f.write('\t\t%f %f %f\n' % tuple(p))
        f.write('\t</points>\n')
        f.write('\t<edges>\n')
        n = 0
        for l in ls:
            f.write('\t\t%d %s\n' % (l,' '.join(map(str,inv[range(n,n+l)])) ) )
            n += l
        f.write('\t</edges>\n')
        f.write('</mesh>\n')

    def write_points(self, f, name):
        ps, color = self.points[name]
        ps = np.unique(ps,axis=0)

        f.write('<mesh color="%d %d %d 1" shader="" name="%s" size="0.6">\n' % tuple(color + [name]) )
        f.write('\t<count>%d 0 0</count>\n' % len(ps) ) #points edges facets
        f.write('\t<points>\n')
        for p in ps:
            f.write('\t\t%f %f %f\n' % tuple(p))
        f.write('\t</points>\n')
        f.write('</mesh>\n')

    def write_subdiv_off(self, f):
        ps, fs, _ = self.meshes[self.mesh_to_subdivide]
        f.write('OFF\n')
        f.write('%d %d 0\n' % (len(ps),len(fs)))
        for p in ps:
            f.write('%f %f %f\n' % tuple(p))
        for fa in fs:
            f.write('%d %s\n' % (len(fa),' '.join(map(str,fa))))
        f.write('\n')

    def read_subdiv_off(self,f):
        f.readline() #discard the first line

        line = f.readline()
        while line[0]=="#":
            line = f.readline() #discard comment lines

        #read teh number of points and faces
        nump,numf,_ = map(int,line.split(' '))
        line = f.readline()

        ps = []
        fs = []
        while nump: #read the points
            #discard comment and empty lines
            if line[0]=="#" or len(line)<3:
                line = f.readline()
                continue

            ps.append(np.array(map(float,line.replace('\n', '').split(' '))))

            line = f.readline()
            nump -= 1


        while numf: #read the faces
            #discard comment and empty lines
            if line[0]=="#" or len(line)<3:
                line = f.readline()
                continue

            fs.append(map(int,line.replace('\n', '').split(' ')[2:]))

            line = f.readline()
            numf -= 1

        color = self.meshes[self.mesh_to_subdivide][2]
        del self.meshes[self.mesh_to_subdivide]
        self.add_mesh(ps,fs,color,name=self.mesh_to_subdivide)

    def show(self,save_and_show=True):

        if self.subdivision_number and self.mesh_to_subdivide:
            with open('subdiv_in.off','wt') as fin:
                self.write_subdiv_off(fin)
            with open('subdiv_in.off','rt') as fin, open('subdiv_out.off','wt') as fout:
                subprocess.call([self.subdiv_path, '%d' % self.subdivision_number],stdin=fin,stdout=fout)
                fout.flush()
            with open('subdiv_out.off','rt') as fout:
                self.read_subdiv_off(fout)

            import os
            os.remove('subdiv_out.off')
            os.remove('subdiv_in.off')

        with open(self.output_file,'wt') as f:
            f.write('<axl>\n')
            for name in self.meshes.keys():
                self.write_mesh(f,name)
            for name in self.polylines.keys():
                self.write_polyline(f,name)
            for name in self.points.keys():
                self.write_points(f,name)
            f.write('</axl>\n')
        if save_and_show:
            #subprocess.call([self.axel_path, 'vis.axl'])
            run = self.axel_path.split(' ')
            run.extend([ self.output_file])
            subprocess.Popen(run)
            time.sleep(3)

    def save(self):
        self.show(False)


Visualization2 = VisualizationAxel

class VisualizationPLY(Visualization):
    def __init__(self):
        super(VisualizationPLY, self).__init__()

    def save(self, mesh_name, fname="vis.ply"):
        ps, fs, color = self.meshes[mesh_name]
        ps,inv = np.unique(ps,axis=0,return_inverse=True)
        nv = len(ps)
        nf = len(fs)
        header = """ply
format ascii 1.0
comment made by Alvaro Fuentes
comment Quad mesh around a skeleton
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar int vertex_index
end_header
""".format(nv,nf)

        with open(fname,"wt") as f:
            f.write(header)
            for pt in ps: f.write("{} {} {}\n".format(*pt))
            for fc in fs:
                fc = [inv[x] for x in fc]
                if len(fc)>len(set(fc)): #check for degenerate quads
                    fc = set(fc)
                f.write("{} {}\n".format(len(fc)," ".join( map(str,fc)) ) )



if __name__=="__main__":

    v = VisualizationVTK()

    ps = np.random.rand(10,3)
    ts = [
        (0,1,2),
        (3,4,5),
        (6,7,8),
        (1,2,9)
    ]
    #v.add_polyline(ps)
    #v.add_points(ps,point_size=3,opacity=.5)
    #v.add_triangle_mesh(ps,ts, opacity=.5)
    v.add_mesh(ps,ts[:2],color=[255,255,0],name='ms')
    v.add_mesh(ps,ts[2:],color=[255,255,0],name='ms2')
    v.add_points(ps[:5],name='ps')
    v.add_points(ps[5:],name='ps')
    v.add_polyline(ps[:5],name='pl')
    v.add_polyline(ps[5:],name='pl')
    v.show()
