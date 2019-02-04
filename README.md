# PySkelton

PySkelton is a Python3 library for skeleton-based modeling. It includes a scaffolding algorithm and anisotropic convolution surfaces.

*Scaffold*: coarse quad mesh that follows the structure of the skeleton, more details in 
> Fuentes Suárez, A. J. & Hubert, E. _Scaffolding skeletons using spherical Voronoi diagrams: Feasibility, regularity and symmetry_ Computer-Aided Design, Elsevier BV, 2018 , 102 , 83-93. [hal-01774909v1](https://hal.inria.fr/hal-01774909v1) DOI [10.1016/j.cad.2018.04.016](https://doi.org/10.1016/j.cad.2018.04.016)

*Anisotropic Convolution Surfaces*: an extension to Convolution Surfaces that adds ellipse-like normal sections, hence anisotropy, around the skeleton. It supports G^1 curves as skeleton, and uses line segments and arcs of circle (circular splines) as skeletal pieces.

## Dependencies

Python:
 * **pyhull:** convex hull computations
 * **numpy:** numerics, linear algebra, vectors
 * **pyroots:** non-derivate root computation (BrentQ method)
 * **swiglpk:** (Gnu Linear Programming Kit) mixed integer linear solver

## Build

In order to use Anisotropic Convolution Surfaces we need to build first the numerical integration code (in C). For that just run `make field_eval_static` in **PySkelton** source folder (`./PySkelton`) to build all the `*.c` files into `field_eval_static.so` shared library that is used by `nformulas.py`. This library implements the numerical integration, and uses *GNU Scientific Library* integration routines from the static library `libgsl.a`.

Failure to build the shared library will prevent the use of anisotropic convolution (`PySkelton.Field`,`PySkelton.Mesher`), but the scaffolding algorithm should be fine.

## Intall python package

The python package can be bulins and installed by running
```shell
python setup.py sdist
python setup.py install
```

