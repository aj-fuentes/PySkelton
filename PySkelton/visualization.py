import subprocess
import collections

import numpy as np
import numpy.linalg as nla
import time

_colors = {'indigo': (75, 0, 130), 'gold': (255, 215, 0), 'firebrick': (178, 34, 34), 'indianred': (205, 92, 92), 'yellow': (255, 255, 0), 'darkolivegreen': (85, 107, 47), 'darkseagreen': (143, 188, 143), 'slategrey': (112, 128, 144), 'darkslategrey': (47, 79, 79), 'mediumvioletred': (199, 21, 133), 'mediumorchid': (186, 85, 211), 'chartreuse': (127, 255, 0), 'mediumblue': (0, 0, 205), 'black': (0, 0, 0), 'springgreen': (0, 255, 127), 'crimson': (220, 20, 60), 'lightsalmon': (255, 160, 122), 'brown': (165, 42, 42), 'turquoise': (64, 224, 208), 'lightseagreen': (32, 178, 170), 'cyan': (0, 255, 255), 'silver': (192, 192, 192), 'skyblue': (135, 206, 235), 'gray': (128, 128, 128), 'darkturquoise': (0, 206, 209), 'goldenrod': (218, 165, 32), 'darkgreen': (0, 100, 0), 'darkviolet': (148, 0, 211), 'darkgray': (169, 169, 169), 'lime': (0, 255, 0), 'lightpink': (255, 182, 193), 'teal': (0, 128, 128), 'darkmagenta': (139, 0, 139), 'lightgoldenrodyellow': (250, 250, 210), 'lavender': (230, 230, 250), 'yellowgreen': (154, 205, 50), 'thistle': (216, 191, 216), 'violet': (238, 130, 238), 'navy': (0, 0, 128), 'dimgrey': (105, 105, 105), 'orchid': (218, 112, 214), 'blue': (0, 0, 255), 'ghostwhite': (248, 248, 255), 'honeydew': (240, 255, 240), 'cornflowerblue': (100, 149, 237), 'purple': (128, 0, 128), 'darkkhaki': (189, 183, 107), 'mediumpurple': (147, 112, 219), 'cornsilk': (255, 248, 220), 'red': (255, 0, 0), 'bisque': (255, 228, 196), 'darkcyan': (0, 139, 139), 'khaki': (240, 230, 140), 'wheat': (245, 222, 179), 'deepskyblue': (0, 191, 255), 'rebeccapurple': (102, 51, 153), 'darkred': (139, 0, 0), 'steelblue': (70, 130, 180), 'aliceblue': (240, 248, 255), 'lightslategrey': (119, 136, 153), 'gainsboro': (220, 220, 220), 'mediumturquoise': (72, 209, 204), 'floralwhite': (255, 250, 240), 'coral': (255, 127, 80), 'aqua': (0, 255, 255), 'lightcyan': (224, 255, 255), 'darksalmon': (233, 150, 122), 'beige': (245, 245, 220), 'azure': (240, 255, 255), 'lightsteelblue': (176, 196, 222), 'oldlace': (253, 245, 230), 'greenyellow': (173, 255, 47), 'fuchsia': (255, 0, 255), 'olivedrab': (107, 142, 35), 'mistyrose': (255, 228, 225), 'sienna': (160, 82, 45), 'lightcoral': (240, 128, 128), 'orangered': (255, 69, 0), 'navajowhite': (255, 222, 173), 'slategray': (112, 128, 144), 'palegreen': (152, 251, 152), 'burlywood': (222, 184, 135), 'seashell': (255, 245, 238), 'mediumspringgreen': (0, 250, 154), 'royalblue': (65, 105, 225), 'papayawhip': (255, 239, 213), 'blanchedalmond': (255, 235, 205), 'peru': (205, 133, 63), 'aquamarine': (127, 255, 212), 'white': (255, 255, 255), 'darkslategray': (47, 79, 79), 'tomato': (255, 99, 71), 'ivory': (255, 255, 240), 'darkgoldenrod': (184, 134, 11), 'lawngreen': (124, 252, 0), 'chocolate': (210, 105, 30), 'orange': (255, 165, 0), 'forestgreen': (34, 139, 34), 'maroon': (128, 0, 0), 'olive': (128, 128, 0), 'mintcream': (245, 255, 250), 'antiquewhite': (250, 235, 215), 'dimgray': (105, 105, 105), 'hotpink': (255, 105, 180), 'moccasin': (255, 228, 181), 'limegreen': (50, 205, 50), 'saddlebrown': (139, 69, 19), 'grey': (128, 128, 128), 'darkslateblue': (72, 61, 139), 'lightskyblue': (135, 206, 250), 'deeppink': (255, 20, 147), 'plum': (221, 160, 221), 'lightgrey': (211, 211, 211), 'dodgerblue': (30, 144, 255), 'slateblue': (106, 90, 205), 'sandybrown': (244, 164, 96), 'magenta': (255, 0, 255), 'tan': (210, 180, 140), 'rosybrown': (188, 143, 143), 'whitesmoke': (245, 245, 245), 'lightblue': (173, 216, 230), 'palevioletred': (219, 112, 147), 'mediumseagreen': (60, 179, 113), 'linen': (250, 240, 230), 'darkorange': (255, 140, 0), 'powderblue': (176, 224, 230), 'seagreen': (46, 139, 87), 'snow': (255, 250, 250), 'mediumslateblue': (123, 104, 238), 'midnightblue': (25, 25, 112), 'paleturquoise': (175, 238, 238), 'palegoldenrod': (238, 232, 170), 'pink': (255, 192, 203), 'darkorchid': (153, 50, 204), 'salmon': (250, 128, 114), 'lightslategray': (119, 136, 153), 'lemonchiffon': (255, 250, 205), 'lightgreen': (144, 238, 144), 'lightgray': (211, 211, 211), 'cadetblue': (95, 158, 160), 'lightyellow': (255, 255, 224), 'lavenderblush': (255, 240, 245), 'darkblue': (0, 0, 139), 'mediumaquamarine': (102, 205, 170), 'green': (0, 128, 0), 'blueviolet': (138, 43, 226), 'peachpuff': (255, 218, 185), 'darkgrey': (169, 169, 169)}

#colors
blue        = _colors['blue']
cyan        = _colors['cyan']
green       = _colors['green']
magenta     = _colors['magenta']
dark_yellow = _colors['goldenrod']
yellow      = _colors['yellow']
dark_green  = _colors['darkgreen']
red         = _colors['red']
black       = _colors['black']
gray        = _colors['gray']

default_palette = [red,yellow,green,cyan,blue,magenta]
pastel_palette = [_colors[k] for k in ["pink","peachpuff","lightgoldenrodyellow","palegreen","paleturquoise","lightsteelblue","thistle"]]
dark_palette = [[(x)/2 for x in color] for color in default_palette]

skel_palette = ["crimson","green","blue","gold","deeppink","darkturquoise","slategray"]

def get_axel_visualization(axel_path="/user/afuentes/home/Programs/miniconda3/bin/axl"):
    # return VisualizationVPython()
    return VisualizationAxel(axel_path)

def _verify_color(color):
    if isinstance(color,str):
        return _colors[color]
    else:
        return color

class Visualization(object):

    def __init__(self):

        self.meshes = {} # name -> points,facets,color
        self.normals = {} #name -> normals for each point
        self.colors = {} #name -> colors for each point
        self.polylines = {} # name -> points,lens,color
        self.points = {} # name -> points,color

    def add_mesh(self, mesh_points, mesh_facets, color=[0,0,255], name=None):
        color = _verify_color(color)
        if name is None:
            name = "Mesh%d" % len(self.meshes)
        if name in list(self.meshes.keys()):
            n = len(self.meshes[name][0])
            self.meshes[name][0].extend(mesh_points)
            fs = list(map(list,mesh_facets))
            for f in fs:
                for i in range(len(f)):
                    f[i] += n
            self.meshes[name][1].extend(fs)
        else:
            self.meshes[name] = (list(mesh_points), list(mesh_facets), list(color) )

    def add_normals(self, normals, name):
        if name in self.normals:
            self.normals[name].extend(normals)
        else:
            self.normals[name] = list(normals)

    def add_colors(self, colors, name):
        assert len(colors[0])==3,"colors must be a list of rgb triplets"
        if name in self.colors:
            self.colors[name].extend(colors)
        else:
            self.colors[name] = list(colors)

    def add_polyline(self, poly_points, color=[255,0,0], name=None):
        color = _verify_color(color)
        if name is None:
            name = "Polyline%d" % len(self.polylines)
        if name in list(self.polylines.keys()):
            self.polylines[name][0].extend(list(poly_points))
            self.polylines[name][1].append(len(poly_points))
        else:
            self.polylines[name] = (list(poly_points), [len(poly_points)], list(color) )

    def add_points(self, points, color=[0,255,0], name=None):
        color = _verify_color(color)
        if name is None:
            name = "Points%d" % len(self.points)

        if name in list(self.points.keys()):
            self.points[name][0].extend(list(points))
        else:
            self.points[name] = (list(points), list(color) )

    def show(self):
        raise NotImplementedError()

    def compute_normals(self):
        for name in self.meshes:
            data = collections.defaultdict(lambda:[])
            ps,fs = self.meshes[name][0],self.meshes[name][1]
            for f_ in fs:
                for i in range(len(f_)):
                    A = ps[f_[i]]
                    B = ps[f_[(i+1)%len(f_)]]
                    C = ps[f_[(i-1)%len(f_)]]

                    u0 = np.cross(B-A,C-A)
                    data[f_[i]].append(u0/nla.norm(u0))

            point_normals = np.zeros(3*len(ps)).reshape(-1,3)
            for i in data:
                ns = np.array(data[i])
                u0 = np.mean(ns,axis=0)
                point_normals[i] = u0/nla.norm(u0)
            self.add_normals(point_normals,name=name)

class VisualizationAxel(Visualization):
    """docstring for Visualization2"""
    def __init__(self, axel_path="/user/afuentes/home/Programs/miniconda3/bin/axl",output_file="vis.axl"):
        super(VisualizationAxel, self).__init__()

        self.output_file = output_file
        self.axel_path = axel_path
        #old axel path:
        #"env LD_LIBRARY_PATH=/home/afuentes/Programs/Axel/lib64:/home/afuentes/Programs/Axel/lib:/user/afuentes/home/Work/Building/qt5.9-install/5.9.2/gcc_64/lib/ /home/afuentes/Programs/Axel/bin/axel"

    def write_mesh(self, f, name):
        ps, fs, color = self.meshes[name]
        # print("Number of points {}".format(len(ps)))

        # ps,inv = np.unique(ps,axis=0,return_inverse=True)
        # print(inv)


        f.write('<mesh color="%d %d %d 1" shader="" name="%s" size="0.05">\n' % tuple(color + [name]) )
        f.write('\t<count>%d 0 %d</count>\n' % (len(ps),len(fs)) ) #points edges facets
        f.write('\t<points>\n')
        for p in ps:
            f.write('\t\t%f %f %f\n' % tuple(p))
        f.write('\t</points>\n')

        if name in self.normals:
            ns = self.normals[name]
            # print("Number of normals {}".format(len(ns)))
            f.write('\t<normals>\n')
            for i in range(len(ps)):
                # f.write('\t\t%f %f %f\n' % tuple(ns[inv[i]]))
                f.write('\t\t%f %f %f\n' % tuple(ns[i]))
            f.write('\t</normals>\n')

        if name in self.colors:
            cs = self.colors[name]
            f.write('\t<colors>\n')
            for i in range(len(ps)):
                f.write('\t\t%d %d %d\n' % tuple(cs[i]))
            f.write('\t</colors>\n')

        #f.write('\t<edges></edges>\n')
        f.write('\t<faces>\n')
        for fa_ in fs:
            # fa = [inv[x] for x in fa_]
            fa = fa_
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
            f.write('\t\t%d %s\n' % (l,' '.join(map(str,inv[list(range(n,n+l))])) ) )
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

    def show(self,save_and_show=True):

        with open(self.output_file,'wt') as f:
            f.write('<axl>\n')
            for name in list(self.meshes.keys()):
                self.write_mesh(f,name)
            for name in list(self.polylines.keys()):
                self.write_polyline(f,name)
            for name in list(self.points.keys()):
                self.write_points(f,name)
            # f.write("<view>\n")
            # f.write("\t<color>1 1 1</color>\n")
            # f.write("\t<gradient>off</gradient>\n")
            # f.write("\t<axis>off</axis>\n")
            # f.write("\t<mode>perspective</mode>\n")
            # f.write('\t<camera>\n')
            # f.write('\t\t<position y="0" z="23.2385" x="0"/>\n')
            # f.write('\t\t<focal y="0" z="0" x="0"/>\n')
            # f.write('\t\t<viewUp y="1" z="0" x="0"/>\n')
            # f.write('\t\t<viewAngle>30</viewAngle>\n')
            # f.write('\t</camera>\n')
            # f.write("</view>\n")
            f.write('</axl>\n')
        if save_and_show:
            #subprocess.call([self.axel_path, 'vis.axl'])
            run = self.axel_path.split(' ')
            run.extend([ self.output_file])
            subprocess.Popen(run)
            time.sleep(3)

    def save(self,fname=None):
        fout = self.output_file
        if fname:
            self.output_file = fname
        self.show(False)
        self.output_file = fout

Visualization2 = VisualizationAxel

if __name__=="__main__":

    v = VisualizationAxel()

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
    v.compute_normals()
    v.show()
