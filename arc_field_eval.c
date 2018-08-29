#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>


//dot product
#define dot(A,B) ((A)[0]*(B)[0] + (A)[1]*(B)[1] + (A)[2]*(B)[2])


/**
  Numerical computation of the integral for a segment piece of the skeleton
**/
struct arc_integrand_params {
    double l; //length
    double r; //circle radius
    double R; //kernel radius
    double XCu; //dot product (X-C).u
    double XCv; //dot product (X-C).v
    double XCuv; //dot product (X-C).(u x v)
    double *a, *b, *c; //radii for interpolation
};

/**
  Numerical computation of the integral for a segment piece of the skeleton
**/
double arc_integrand_function(double t, void * ps) {

    struct arc_integrand_params * params = (struct arc_integrand_params *) ps;

    double r = params->r;
    double l = params->l;
    double lt = l - t;
    double R = params->R;

    double st = sin(t/r);
    double ct = cos(t/r);

    double XCu = params->XCu;
    double XCv = params->XCv;
    double XCuv = params->XCuv;

    double a = l * (-XCu*st + XCv*ct    ) / (params->a[0] * lt + params->a[1] * t);
    double b = l * ( XCuv               ) / (params->b[0] * lt + params->b[1] * t);
    double c = l * ( XCu*ct + XCv*st - r) / (params->c[0] * lt + params->c[1] * t);
    double d = 1.0e0 - (a * a + b * b + c * c) / (R * R);
    if (d < 0.0e0) return 0.0e0;
    else return d * d * d;
}

double arc_compact_field_eval(double *X, double *C, double r, double *u, double *v, double phi, double *a, double *b, double *c, double R, unsigned int n, double max_err) {
    //N=cross product u x v
    double N[3]  = {
        u[1]*v[2] - u[2]*v[1],
       -u[0]*v[2] + u[2]*v[0],
        u[0]*v[1] - u[1]*v[0]
    };

    double XC[3] = {X[0] - C[0], X[1] - C[1], X[2] - C[2]};

    double XCu   =  dot(XC,u);
    double XCv   =  dot(XC,v);
    double XCuv  =  dot(XC,N);

    double l = r*phi;

    //define the parameters for the integrand in GSL
    struct arc_integrand_params params = {l, r, R, XCu, XCv, XCuv, a, b, c};

    gsl_function F;
    //define the integrand function for GSL
    F.function = &arc_integrand_function;
    //setup params
    F.params = &params;

    //define workspace for GSL (to higher precision)
    gsl_integration_workspace * ws = gsl_integration_workspace_alloc((size_t)n);

    double val;
    double err;

    //integrate
    int res = gsl_integration_qag (&F, 0.0e0, l, max_err, max_err, (size_t)n, GSL_INTEG_GAUSS61, ws, &val, &err);

    //free GSL workspace
    gsl_integration_workspace_free(ws);

    return val;
}

