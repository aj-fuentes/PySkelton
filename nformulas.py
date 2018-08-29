import ctypes
import numpy as np

#parameter types for vectors
vector_3d_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1,shape=(3,),flags='C_CONTIGUOUS')
#parameter types for matrices
matrix_3d_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=2,shape=(3,3),flags='C_CONTIGUOUS')
#parameter types for radii array
vector_2d_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1,shape=(2,),flags='C_CONTIGUOUS')


################ NEW LIB ##############################

field_eval_path = "/user/afuentes/home/Work/Convolution/code/python/package/field_eval.so"
field_eval_lib = ctypes.cdll.LoadLibrary(field_eval_path)

#declare the eval fucntion in the library
compact_field_eval = field_eval_lib.compact_field_eval
arc_compact_field_eval = field_eval_lib.arc_compact_field_eval

#declare the parameters
compact_field_eval.argtypes = [
    vector_3d_double, #X
    vector_3d_double, #P
    vector_3d_double, #T
    vector_3d_double, #N
    ctypes.c_double,  #l
    vector_2d_double, #a
    vector_2d_double, #b
    vector_2d_double, #c
    ctypes.c_double,  #R
    ctypes.c_uint,    #n
    ctypes.c_double,  #maxerror
    ]
#declare the parameters
arc_compact_field_eval.argtypes = [
    vector_3d_double, #X
    vector_3d_double, #C
    ctypes.c_double,  #r
    vector_3d_double, #u
    vector_3d_double, #v
    ctypes.c_double,  #phi
    vector_2d_double, #a
    vector_2d_double, #b
    vector_2d_double, #c
    ctypes.c_double,  #R
    ctypes.c_uint,    #n
    ctypes.c_double,  #maxerror
    ]

#declare return type
compact_field_eval.restype = ctypes.c_double
arc_compact_field_eval.restype = ctypes.c_double

#declare documentation
compact_field_eval.__doc__ = "double compact_field_eval(double *X, double *P, double *T, double *N, double l, double *a, double *b, double *c, double R, size_t n, double max_err)"
arc_compact_field_eval.__doc__ = "double arc_compact_field_eval(double *X, double *C, double r, double *u, double *v, double phi, double *a, double *b, double *c, double R, size_t n, double max_err)"

