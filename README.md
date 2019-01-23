# PySkelton

PySkelton is a Python3 library for skeleton-based modeling. It includes a scaffolding algorithm and anisotropic convolution surfaces.

*Scaffold*: coarse quad mesh that follows the structure of the skeleton for more details in [1](https://hal.inria.fr/hal-01774909v1)

*Anisotropic Convolution Surfaces*: an extension to Convolution Surfaces that adds ellipse-like normal sections, hence anisotropy, around the skeleton. It supports G^1 curves as skeleton, and uses line segments and arcs of circle (circular splines) as skeletal pieces.

## Dependencies

Python:
 * **pyhull:** convex hull computations
 * **numpy:** numerics, linear algebra, vectors
 * **pyroots:** non-derivate root computation (BrentQ method)
 * **other**

C:
 * **GSL:** (Gnu Scientific Library) numerical integration

System:
 * **GLPK:** (Gnu Linear Programming Kit) mixed integer linear solver

## Build

In order to use Anisotropic Convolution Surfaces we need to build first the numerical integration code (in C). For that just run `make` to build all the `*.c` files into `field\_eval.so` shared library that is used by `nformulas.py`.


