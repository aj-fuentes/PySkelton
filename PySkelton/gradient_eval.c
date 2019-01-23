#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>


//dot product
#define dot(A,B) ((A)[0]*(B)[0] + (A)[1]*(B)[1] + (A)[2]*(B)[2])
#define max(a,b) ((a) > (b) ? (a) : (b))


/**
  Numerical computation of the integral for a segment piece of the skeleton
**/
struct integrand_gradient_params {
    double l; //length
    double R; //kernel radius
    double XPT; //dot product (X-P).T
    double XPN; //dot product (X-P).N
    double XPB; //dot product (X-P).B
    double *T,*N,*B; //frame vectors
    double *a, *b, *c; //radii for interpolation
    double *th; //angle for interpolation
    int    deriv; //derivative to compute
    // 0->a0,1->a1,2->b0,3->b1,4->c0,5->c1,6->theta0,7->theta1
};

/**
  Numerical computation of the integral for a segment piece of the skeleton
**/
double integrand_gradient(double t, void * ps) {

    struct integrand_gradient_params * params = (struct integrand_gradient_params *) ps;


    double l = params->l;
    double lt = l - t;
    double R = params->R;

    double th = (params->th[0]*lt + params->th[1]*t)/l;
    double _cos = cos(th);
    double _sin = sin(th);

    double XPN =  params->XPN * _cos + params->XPB * _sin;
    double XPB = -params->XPN * _sin + params->XPB * _cos;


    double da = l / (params->a[0] * lt + params->a[1] * t);
    double db = l / (params->b[0] * lt + params->b[1] * t);
    double dc = l / (params->c[0] * lt + params->c[1] * t);

    double a = (params->XPT - t) * da;
    double b = (            XPN) * db;
    double c = (            XPB) * dc;

    double d = 1.0e0 - (a * a + b * b + c * c) / (R * R);
    if (d < 0.0e0) return 0.0e0;

    int deriv = params->deriv;

    double Ni =  params->N[deriv] * _cos + params->B[deriv] * _sin;
    double Bi = -params->N[deriv] * _sin + params->B[deriv] * _cos;

    double val = a * da * (params->T[deriv]) + b * db * (Ni) + c * dc * (Bi);

    return d * d * val;
}

double compact_gradient_eval(double *X, double *P, double *T, double *N, double l, double *a, double *b, double *c, double* th, double max_r, double R, int deriv, unsigned int n, double max_err) {
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
    struct integrand_gradient_params params = {l, R, XPT, XPN, XPB, T, N, B, a, b, c, th, deriv};

    gsl_function F;
    //define the integrand function for GSL
    F.function = &integrand_gradient;
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

    // printf("%.15f\n", val);

    return -6.0e0/(R*R) * val/R*2.1875e0; //adjust to get value 1.0 at extremities
}
