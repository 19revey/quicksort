#ifndef SYMS_H
#define SYMS_H


#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>
#include"particle.h"


#define CEILING(X) (X-(int)(X) > 0 ? (int)(X+1) : (int)(X))
#define delta_fun(r)  (fabs(r) <= 2 ? ((1. + cos(M_PI*(r)/2.)) * 0.25) : 0.0)
#define min(x,y)  (x<y? x:y)
#define max(x,y)  (x>y? x:y)
class sph_param{
protected:
    int    nframes;       // number of output snapshot
    int    npframe;       // number of steps between two consecutive snapshots
    double hh;            // smoothing length
    double h;             // initial gap
    double Wh0;           // W(h0/h)
    double dt;            // time step
    double rho0;          // initial fluid density
    double k;             // fluid conductivity
    double mu;            // fluid dynamic viscosity
    double nu;            // fluid kinematic viscosity
    double g;             // gravity constant
    double dpdx;          // pressure gradient
    double Utop,Vtop;     // velocity at the top wall
    int    MAX_LISTELEM;  // max length of the neighboring particle list
    int    factor;        // ratio of the effective circle radius to h.
    double xc,yc,rad;     // cylinder center and radius
    double alpha;         // thermal diffusivity
    double Ts;            // sphere temperature
    double Tf0;            // fluid initial temperature
    double Tw_up,Tw_bot;  // wall termperature
    double cspeed;        // speed of sound
    double Vpx, Vpy;      // particle velocity
    double XMIN,XMAX,YMIN,YMAX;
    int    type;  //0: no particles; 1: fixed; 2: moving; 3: non-ibm BC fixed; 4: non-ibm BC moving
    int    Bnodes;
    int    PLACEMENT,SEARCH,PRESSURE,TENSILE;
    double Gr,Re,Pr;
    double Tref,beta,Lref,Vref;

public:
    char   fname0[40];     // project name

    sph_param();
    void read_params(char *inFile);
    double kernelW(int factor, double dis, double h);
    void print_info();
    ~sph_param(){};
};

void sph_param::print_info()
{
    char fname[40];
    sprintf(fname,"%s.info",fname0);

    FILE *fl=fopen(fname,"w");

    printf("Box size [%.4lf %.4lf] X [%.4lf %.4lf], Cyl Rad=%.4lf\n",
            XMIN,XMAX,YMIN,YMAX,rad);

    printf("SPH hh=%.4e h=1.5hh factor=%d Cspeedfactor=%.2lf\n",hh,factor,cspeed);

    printf("Problem Type (1-2: ibm; 3-4: non-ibm)=%1d PLACEMENT (0:regualar) =%1d SEARCH(0: spatial)=%1d PRESSURE(0: Morris)=%d TENSILE(0: no artifical P)=%1d\n",
            type,PLACEMENT,SEARCH,PRESSURE,TENSILE);

    printf("Fluid density=%.4e  viscosity=%.4e  theraml diffusivity (alpha)=%.4e  expansion coef (beta)=%.4e\n",
            rho0,mu,alpha,beta);

    printf("Flow Re=%.4lf Gr=%.4lf Pr=%.4lf\n",Re,Gr,Pr);
    fclose(fl);
}

double sph_param::kernelW(int factor, double dis, double h)
{

    double temp=0;
    if(factor==2){
        if(dis<=h){
            double norm=10./(7*M_PI*h*h),s=dis/h;
            return (norm*(1-1.5*s*s+.75*s*s*s));
        }
        if (dis<=2*h){
            double norm=10./(7*M_PI*h*h),s=dis/h;
            return (norm*(2-3*s+1.5*s*s-.25*s*s*s));
        }
        return 0.;
    }
    else{
        if(dis<=h){
            double s=dis/h;
            return 7./(487*M_PI*h*h)*(-10*s*s*s*s*s+30*s*s*s*s-60*s*s+66);
        }
        if (dis<=2*h){
            double s=dis/h;
            return 7./(487*M_PI*h*h)*(5*s*s*s*s*s-45*s*s*s*s+150*s*s*s-210*s*s+75*s+51);
        }
        if(dis<=3*h){
            double s=dis/h;
            return 7./(487*M_PI*h*h)*(-s*s*s*s*s+15*s*s*s*s-90*s*s*s+270*s*s-405*s+243);
        }
        return 0.;
    }
}

sph_param::sph_param()
{
    strcpy(fname0,"sphTest");
    nframes = 20;
    npframe = 500;
    rad     =0.2;// 0.02;
    dt      = 0.1;
    hh      = 1.5625e-2;//1.5625e-2;//1.5625e-3;
    h       = hh*1.5;//.25e-2;

    cspeed  = 10;//*vmax5.77e-4;
    rho0    = 1000;
    mu      = 0.001;//0.001;
    g       = 9.8;
    dpdx    = 1.5e-4;//1.5e-4;//1.5e-7;//2e-4;
    Utop=0;
    Vtop=0;
    Tw_up=0;
    Tw_bot=0;
    Tf0=0;
    Ts=1;
    MAX_LISTELEM=151;

    PLACEMENT=1;
    SEARCH=1;
    PRESSURE=0; // 0: p=c^2rho; 1: p=c^2(rho-rho0)
    TENSILE=0;

    factor=3;
    XMIN=0.;
    XMAX=1.0;//0.1;
    YMIN=0;
    YMAX=1;//0.1;
    type =1;
    Vpx=0;
    Vpy=0;
    // for heat convection
    Pr=10.0;
    alpha=(mu/rho0)/(Pr);//*params->rho0;
    // Grashof number: Gr=rho^2*g*beta*Tc*Lc^3/mu^2
    // bouyance:  rho*beta*(T-T0)*g
    // free convection: Re=1=rho0*Vref*rad/mu;  Vref=mu/(rho0*rad)

    //  params->Re=sqrt(param->Gr);
    //    params->Vref=params->Re*params->mu/(params->rho0*2*params->rad);
    // params->cspeed=100.*params->Vref;

    // Given Gr or Ra
    Gr=100;
    Tref=Ts-Tf0;

    beta=Gr*mu*mu/(8*rho0*rho0*g*Tref*rad*rad*rad);
    double Vref=sqrt(rho0*beta*Tref*2*rad);
    printf("beta=%lf cspeed=%lf vref=(%lf %lf) tmax=(conv: %lf  diff %lf)\n",beta,cspeed,Vref,Vref,
           hh/cspeed,0.125*hh*hh*rho0/mu);

    //params->Vref=Vref;//params->Re*params->mu/(params->rho0*2*params->rad);
    //  params->cspeed=10.*params->Vref;
    Wh0=kernelW(factor,hh,h);
}


#endif // SYMS_H

