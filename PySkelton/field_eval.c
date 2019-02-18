#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "gsl/gsl_sf_bessel.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_block.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_math.h"
#include "gsl/gsl_roots.h"


//dot product
#define dot(A,B) ((A)[0]*(B)[0] + (A)[1]*(B)[1] + (A)[2]*(B)[2])
#define max(a,b) ((a) > (b) ? (a) : (b))


/**
  Numerical computation of the integral for a segment piece of the skeleton
**/
struct integrand_params {
    double l; //length
    double R; //kernel radius
    double XPT; //dot product (X-P).T
    double XPN; //dot product (X-P).N
    double XPB; //dot product (X-P).B
    double *a, *b, *c; //radii for interpolation
    double *th; //angle for interpolation
};

/**
  Numerical computation of the integral for a segment piece of the skeleton
**/
double integrand_function(double t, void * ps) {

    struct integrand_params * params = (struct integrand_params *) ps;

    double l = params->l;
    double lt = l - t;
    double R = params->R;

    double th = (params->th[0]*lt + params->th[1]*t)/l;
    double _cos = cos(th);
    double _sin = sin(th);

    double XPN =  params->XPN * _cos + params->XPB * _sin;
    double XPB = -params->XPN * _sin + params->XPB * _cos;

    double a = l * (params->XPT - t) / (params->a[0] * lt + params->a[1] * t);
    double b = l * (XPN) / (params->b[0] * lt + params->b[1] * t);
    double c = l * (XPB) / (params->c[0] * lt + params->c[1] * t);
    double d = 1.0e0 - (a * a + b * b + c * c) / (R * R);
    if (d < 0.0e0) return 0.0e0;
    else return d * d * d / a;
}

double compact_field_eval(double *X, double *P, double *T, double *N, double l, double *a, double *b, double *c, double* th, double max_r, double R, unsigned int n, double max_err) {
    //B=cross product T x N
    double B[3]  = {
        T[1]*N[2] - T[2]*N[1],
       -T[0]*N[2] + T[2]*N[0],
        T[0]*N[1] - T[1]*N[0]
    };

    double XP[3] = {X[0] - P[0], X[1] - P[1], X[2] - P[2]};

    double XPT   =  dot(XP,T);
    double XPN   =  dot(XP,N);
    double XPB   =  dot(XP,B);

    double t = (XPT>0.0e0)? ((XPT<l)? 0.0e0 : XPT-l) : XPT;

    if((t*t + XPN*XPN + XPB*XPB) > (max_r*max_r*R*R)) return 0.0e0;

    //define the parameters for the integrand in GSL
    struct integrand_params params = {l, R, XPT, XPN, XPB, a, b, c, th};

    gsl_function F;
    //define the integrand function for GSL
    F.function = &integrand_function;
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

    return val/R*2.1875e0; //adjust to get value 1.0 at extremities
}
