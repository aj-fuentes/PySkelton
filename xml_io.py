import xml.etree.ElementTree as ET

import numpy as np

from . import skeleton as sk
from . import field as fl


def to_xml_element(obj,**extra_attribs):
    elem = _makers[type(obj)](obj)
    elem.attrib.update(extra_attribs)
    # ET.dump(elem)
    return elem

def to_xml_document(obj):
    root = to_xml_element(obj)
    return '<?xml version="1.0"?>\n' + ET.tostring(root,encoding="unicode",method="xml")

def parse_xml_document(data):
    root = ET.fromstring(data)
    return parse_xml_element(root)

def parse_xml_element(elem):
    # print("parsing {}".format(ET.tostring(elem,encoding="unicode")))
    return _parsers[elem.tag](elem)

_sub = ET.SubElement
_elem = ET.Element

def _arr_format(arr):
    return " ".join(map(str,arr))

def _segment_to_xml_element(obj):
    xe = _elem("segment",attrib={"l":str(obj.l)})
    _sub(xe,"P").text=_arr_format(obj.P)
    _sub(xe,"v").text=_arr_format(obj.v)
    _sub(xe,"n").text=_arr_format(obj.n)
    return xe

def _parse_segment_xml_element(elem):
    P = np.fromstring(elem.find("P").text,sep=" ")
    v = np.fromstring(elem.find("v").text,sep=" ")
    l = float(elem.attrib["l"])
    n = np.fromstring(elem.find("n").text,sep=" ")
    return sk.Segment(P,v,l,n)

def _arc_to_xml_element(obj):
    xe = _elem("arc",attrib={"r":str(obj.r),"phi":str(obj.phi)})
    _sub(xe,"C").text=_arr_format(obj.C)
    _sub(xe,"u").text=_arr_format(obj.u)
    _sub(xe,"v").text=_arr_format(obj.v)
    return xe

def _parse_arc_xml_element(elem):
    C = np.fromstring(elem.find("C").text,sep=" ")
    u = np.fromstring(elem.find("u").text,sep=" ")
    v = np.fromstring(elem.find("v").text,sep=" ")
    r = float(elem.attrib["r"])
    phi = float(elem.attrib["phi"])
    return sk.Arc(C,u,v,r,phi)

def _g1curve_to_xml_element(obj):
    xe = _elem("g1curve")
    xe.extend(to_xml_element(skel) for skel in obj.skels)
    return xe

def _parse_g1curve_xml_element(elem):
    skels = [parse_xml_element(e) for e in elem]
    return sk.G1Curve(skels)

def _field_to_xml_element(obj):
    attr = {
        "R": str(obj.R),
        "a": _arr_format(obj.a),
        "b": _arr_format(obj.b),
        "c": _arr_format(obj.c),
        "th": _arr_format(obj.th)
        }
    xe = _elem("field",attrib=attr)
    xe.extend([to_xml_element(obj.skel)])
    return xe

def _parse_field_xml_element(elem):
    R = float(elem.attrib["R"])
    skel = parse_xml_element(elem[0])
    a = np.fromstring(elem.attrib["a"],sep=" ")
    b = np.fromstring(elem.attrib["b"],sep=" ")
    c = np.fromstring(elem.attrib["c"],sep=" ")
    th = np.fromstring(elem.attrib["th"],sep=" ")
    return fl.make_field(R,skel,a,b,c,th)

def _multifield_to_xml_element(obj):
    xe = _elem("multifield")
    xe.extend([to_xml_element(field,coeff=str(coeff)) for coeff,field in zip(obj.coeffs,obj.fields)])
    return xe

def _parse_multifield_xml_element(elem):
    fields = [parse_xml_element(e) for e in elem]
    coeffs = [float(e.attrib["coeff"]) for e in elem]
    field = fl.MultiField(fields)
    field.set_coeffs(coeffs)
    return field

_makers = {
    sk.Segment:         _segment_to_xml_element,
    sk.Arc:             _arc_to_xml_element,
    sk.G1Curve:         _g1curve_to_xml_element,
    fl.SegmentField:    _field_to_xml_element,
    fl.ArcField:        _field_to_xml_element,
    fl.G1Field:         _field_to_xml_element,
    fl.MultiField:      _multifield_to_xml_element
}

_parsers = {
    "segment": _parse_segment_xml_element,
    "arc": _parse_arc_xml_element,
    "g1curve": _parse_g1curve_xml_element,
    "field": _parse_field_xml_element,
    "multifield": _parse_multifield_xml_element
}
