#ifndef PARTICLE_H
#define PARTICLE_H

#include"syms.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>
#include"cudasort.cuh"

#define CEILING(X) (X-(int)(X) > 0 ? (int)(X+1) : (int)(X))
#define delta_fun(r)  (fabs(r) <= 2 ? ((1. + cos(M_PI*(r)/2.)) * 0.25) : 0.0)
#define min(x,y)  (x<y? x:y)
#define max(x,y)  (x>y? x:y)



class sph_fluid: public sph_param{

protected:
    int n,nb;
    double mass;
    double *rho,*rhoh;
    double *xp, *yp;
    double *vxh, *vyh;
    double *vx, *vy;
    double *ax, *ay;
    double *fbx,*fby,*hb;
    double *Tf,*Tfh, *drho,*dT;
    int    *solid;
    double *xb,*yb,*vxb,*vyb,*Tb,*Tsb;
    int    **neighborList; /* store neighbor index*/
    double BodyFx,BodyFy,BodyNu;

public:
    void place_particles();
    void alloc_state(int n, int nb, int MAX_LISTELEM);
    int** make2dmemInt(int arraySizeX, int arraySizeY);
    double** make2dmem(int arraySizeX, int arraySizeY);
    void normalize_mass();
    void iterations(double *timeused);
    void plot_velocity_flux(int count, double *flow1, double *flow2);
    void line_temperature(double x0, int *count,double *y, double *vel);
    void line_flux_left_wall(double x0,int *count,double *y,double *flux);
    void gradientKernelW(int factor,double diffx, double diffy, double dis, double h,double *Wx,double *Wy);
    void find_T_and_dT_at_wall();
    double max_vel();
    void find_neighbor_brute(int n,double *xp,double *yp,double dis,int **neighborList,int M,double Lx,double Ly);
 //   extern void find_neighbor_spatial(int n,double *xp,double *yp,double dis,int **neighborList,int M);
    void compute_bforce();
    void solid_movement();
    void compute_time_derivatives();
    double db_da_for_cylinder(double xi,double yi, double xj,double yj, double xc, double yc, double rad);
    void movement_step();
    void compute_DragHeat();
    
    void find_neighbor_spatial(int n,double* xp,double* yp,double xh,int** neighborList,int MAX_LISTELEM);
    void print_frame(FILE *fp_solid);
    ~sph_fluid(){};	
};



void sph_fluid::iterations(double *timeused)
{
    FILE* fp    = fopen(fname0, "w");   //
    FILE* fk;                                 //
    double flow1,flow2;                       //
    FILE *fp1;                                //
    char fname[40];                           //
        sprintf(fname,"%s_drag.dat",fname);       //

    for (int frame = 1; frame < nframes; ++frame) {

        sprintf(fname,"%s.%d.vtk",fname0,frame);
        fp1=fopen(fname,"w");

        print_frame(fp1);
        fclose(fp1);

        plot_velocity_flux(frame,&flow1,&flow2);             /////////////

        find_T_and_dT_at_wall();                              /////////////////
        printf("present result (frame %d ......... flow1=%.5e flow2=%.5e, max-vel=%.5e\n",frame,flow1,flow2,max_vel());
        for (int i = 0; i < npframe; ++i) {

            clock_t start2 = clock ();
            if(SEARCH==0){
                find_neighbor_brute(n,xp,yp,factor*h,
                                    neighborList,MAX_LISTELEM,
                                    XMAX-XMIN,YMAX-YMIN);
            }

            else{
                find_neighbor_cuda(n,xp,yp,factor*h,neighborList,MAX_LISTELEM);
            }
            
 //
             for (int i=0; i<n; i++)
             {
                 if (neighborList[i][MAX_LISTELEM-1]>=MAX_LISTELEM-1)
                 {
                     printf(" The pre-allocated MAX_LISTELEM (%d) is too small, there are (%d) found\n",    MAX_LISTELEM, neighborList[i][MAX_LISTELEM-1]);
                     exit(0);
                 }
             }
 //
            
            clock_t timeElapsed2 = ( clock() - start2 ) / (CLOCKS_PER_SEC/1000);
            timeused[2]=timeused[2]+timeElapsed2;

            
            clock_t start = clock ();    //timer in search
            compute_bforce();                                   ////////////////
            clock_t timeElapsed = ( clock() - start ) / (CLOCKS_PER_SEC/1000);
            timeused[0]=timeused[0]+timeElapsed;
            
            
            clock_t start1 = clock ();
            solid_movement();                                      //////////////////////////
            compute_time_derivatives();                                //////////
            movement_step();                                       /////////////////////////
            clock_t timeElapsed1 = ( clock() - start1 ) / (CLOCKS_PER_SEC/1000);
            timeused[1]=timeused[1]+timeElapsed1;
            
            if(i%40 && (type==3 || type==4)){
                compute_DragHeat();
                sprintf(fname,"%s_drag.dat",fname0);
                fk=fopen(fname,"a");
                fprintf(fk,"%10.4lf  %.5e  %.5e  %.5e\n",(frame*npframe+i)*dt,BodyFx,BodyFy,BodyNu);
                fclose(fk);
            }
        }
    }
    fclose(fk);
    printf("Done....");
    fclose(fp);
}

void sph_fluid::plot_velocity_flux(int count, double *flow1, double *flow2)
{
    char fname[30];
    sprintf(fname,"%s.%d.tem",fname0,count);
    FILE *fp1=fopen(fname,"w");
    sprintf(fname,"%s.%d.fx1",fname0,count);
    FILE *fp2a=fopen(fname,"w");
    sprintf(fname,"%s.%d.fx2",fname0,count);
    FILE *fp2b=fopen(fname,"w");

    int nj=(5*(YMAX-YMIN)/hh),count1,count2a,count2b;

    double *y1=(double*)calloc((nj+1),sizeof(double));
    double *y2a=(double*)calloc((nj+1),sizeof(double));
    double *y2b=(double*)calloc((nj+1),sizeof(double));
    double *yv1=(double*)calloc((nj+1),sizeof(double));
    double *yv2a=(double*)calloc((nj+1),sizeof(double));
    double *yv2b=(double*)calloc((nj+1),sizeof(double));
    line_temperature((XMAX-XMIN)/2.,&count1,y1,yv1);

    line_flux_left_wall(0.5*h,&count2a,y2a,yv2a);
    line_flux_left_wall(2*h,&count2b,y2b,yv2b);

    *flow1=0;
    *flow2=0;
    for(int j=0;j<count1;j++){
        *flow1+=yv1[j];
        fprintf(fp1,"%.5e  %.5e\n",y1[j],yv1[j]);
    }

    for(int j=0;j<count2a;j++){
        *flow2+=yv2a[j];
        fprintf(fp2a,"%.5e  %.5e\n",y2a[j],yv2a[j]);
    }

    for(int j=0;j<count2b;j++){

        fprintf(fp2b,"%.5e  %.5e\n",y2b[j],yv2b[j]);
    }

    free(y1);free(y2a);free(y2b);
    free(yv1);free(yv2a);free(yv2b);
    fclose(fp1);
    fclose(fp2a);
    fclose(fp2b);

}

void sph_fluid::line_temperature(double x0, int *count, double *y, double *vel)
{
    // find particles within x0+hh
    int icount=0;

    for(int i=0;i<n;i++){
        if((xp[i]<=x0+.5*h)&&(xp[i]>=x0-0.5*h)){
            y[icount]=yp[i];
            vel[icount]=Tf[i];
            icount++;
        }
    }
    *count=icount;
}

void sph_fluid::line_flux_left_wall(double x0,int *count,double *y,double *flux)
{
    // find particles within x0+hh
    double Twall=0,beta1=1.5,db_da,dTf;
    int icount=0;

    for(int i=0;i<n;i++){

        if((xp[i]<=x0+.5*h)&&(xp[i]>=x0-.5*h)){
            y[icount]=yp[i];
            // needs neighboring particles
            flux[icount]=0;
            for(int j=0;j<neighborList[i][MAX_LISTELEM-1];j++){
                int jj=neighborList[i][j];
                double dx = xp[i]-xp[jj], dy= yp[i]-yp[jj];
                double dis=sqrt(dx*dx+dy*dy);

                if(type==3 || type==4){
                    if(solid[jj]==1){// only consider left side wall
                        db_da=(-xp[jj])/max(1.e-10,xp[i]);
                        dTf = (Tf[i]-Twall)*min(beta1,1+db_da);
                    }

                    else{
                        dTf=Tf[i]-Tf[jj];
                    }
                }
                else{ // ibm
                    dTf=Tf[i]-Tf[jj];
                }


                // using dT/dx=1/rhoi*mass*sum(Tj-Ti)*dW/dx
                double Wx,Wy;
                gradientKernelW(factor,dx,dy,dis,h,&Wx,&Wy);
                flux[icount]+=1./rho[i]*(-dTf)*Wx*mass;
            }
            icount++;
        }
    }
    *count=icount;

}
void sph_fluid::gradientKernelW(int factor,double diffx, double diffy, double dis, double h,double *Wx,double *Wy)
{
    double temp=0;
    if(factor==2){
        if(dis<=h){
            double norm=10./(7*M_PI*h*h*h),s=dis/h;
            temp=norm*(-3*s+2.25*s*s);
        }
        else if (dis<=2*h){
            double norm=10./(7*M_PI*h*h*h),s=dis/h;
            temp=norm*(-3+3*s-0.75*s*s);
        }
    }
    else{
        if(dis<=h){
            double s=dis/h;
            temp= 7./(487*M_PI*h*h*h)*(-50*s*s*s*s+120*s*s*s-120*s);
        }
        else if (dis<=2*h){
            double s=dis/h;
            temp=7./(487*M_PI*h*h*h)*(25*s*s*s*s-180*s*s*s+450*s*s-420*s+75);
        }
        else if(dis<=3*h){
            double s=dis/h;
            temp=7./(487*M_PI*h*h*h)*(-5*s*s*s*s+60*s*s*s-270*s*s+540*s-405);
        }
    }
    *Wx=temp*diffx/dis; *Wy=temp*diffy/dis;

}

void sph_fluid::find_T_and_dT_at_wall()
{
    int nr=nb;
    char fname[30];
    sprintf(fname,"%s_scalar.dat",fname0);
    FILE *fp1=fopen(fname,"w");

    double *T=(double*)calloc((nr+1),sizeof(double));
    double *dTdr=(double*)calloc((nr+1),sizeof(double));
    double Xc=xc,Yc=yc;
    double R=rad;//+params->h*params->factor;

    for(int i=0;i<=nr;i++){
        double theta=i*(2*M_PI/nr);
        double xi=Xc-R*cos(theta);
        double yi=Yc+R*sin(theta);
        T[i]=0;
        dTdr[i]=0;

        for(int jj=0;jj<n;jj++){
            double diffx=xi-xp[jj];
            double diffy=yi-yp[jj];

            double dis=sqrt(diffx*diffx+diffy*diffy);
            double W=kernelW(factor,dis,h);
            double Wx,Wy;
            gradientKernelW(factor,diffx,diffy,dis,h,&Wx,&Wy);
            double dV=mass/rho[jj];
            T[i]+=Tf[jj]*W*dV;
            dTdr[i]+=Tf[jj]*(Wx*diffx+Wy*diffy)*dV;
        }
        dTdr[i]/=(1.-0);
    }
    for(int i=0;i<=nr;i++){
        fprintf(fp1,"%.2lf  %.5e  %.5e\n",i*(360.0/nr),T[i],-dTdr[i]);
    }
    free(T);free(dTdr);
    fclose(fp1);
}

double sph_fluid::max_vel()
{
    double maxV=0;
    for(int i=0;i<n;i++){
        double v2=vx[i]*vx[i]+vy[i]*vy[i];
        if(v2>maxV){
            maxV=v2;
        }
    }
    return sqrt(maxV);
}

void sph_fluid::find_neighbor_brute(int n,double *xp,double *yp,double dis,int **neighborList,
                         int MAX_LISTELEM,double Lx,double Ly)
{
    double max=100000,Dis2=dis*dis;
    int pos=-1;

    double **PointSet=make2dmem(2,n);
    int *count=(int*)calloc(n,sizeof(int));

    for(int i=0;i<n;i++){
        PointSet[0][i]=xp[i];
        PointSet[1][i]=yp[i];
    }

    for(int i=0;i<n;i++){
        for(int j=0;j<MAX_LISTELEM;j++){
            neighborList[i][j]=0;
        }
    }
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            double dx=(PointSet[0][j]-PointSet[0][i]);
            /*  if(dx>Lx/2.){
             dx-=Lx;
             }
             else if(dx<-Lx/2){
             dx+=Lx;
             }
             */
            double dy=(PointSet[1][j]-PointSet[1][i]);
            /*if(dy>Ly/2){
             dy-=Ly;
             }
             else if(dy<-Ly/2.){
             dy+=Ly;
             }*/
            double t=dx*dx+dy*dy;

            if(t<Dis2 && t>1.0e-20){
                neighborList[i][count[i]]=j;
                neighborList[j][count[j]]=i;
                count[i]++;
                count[j]++;

                if(max(count[i],count[j])>=MAX_LISTELEM-1){
                    printf("number of neighbors for particle %d located at (%.4e %.4e) exceeds the allocated MAX_LISTELEM.\n",i,xp[i],yp[i]);
                    exit(0);
                }
            }
        }
    }
    for(int i=0;i<n;i++){
        neighborList[i][MAX_LISTELEM-1]=count[i];
    }

    for(int i=0;i<2;i++){
        free(PointSet[i]);
    }
    free(PointSet);
}


void sph_fluid:: compute_bforce()
{
    if(type==0 || type==3 || type==4){ //no particle or nonIBM
        return;
    }
    /*
     if(params->type==0 || params->type==3 || params->type==4){ //no particle or nonIBM
     if(params->type==4){
     params->xc+=params->Vpx*params->dt;
     if(params->xc>params->XMAX){
     params->xc-=params->XMAX;
     }
     else if(params->xc<params->XMIN){
     params->xc+=params->XMAX;
     }
     params->yc+=params->Vpy*params->dt;
     }
     return;
     }
     */
    int k=0;
    int idb[1000],idf[1000];
    double dis,delta;
    double h2 = h*h;

    // compute velocity at the boundary nodes
    for (int ib = 0; ib < nb; ib++) {
        vxb[ib]=0;
        vyb[ib]=0;
        Tb[ib]=0;
        for (int i = 0; i <n; i++) {
            double dx = xp[i]-xb[ib];
            double dy = yp[i]-yb[ib];
            /*if(dx>(params->XMAX-params->XMIN)/2.){
             dx-=params->XMAX-params->XMIN;
             }
             else if(dx<-(params->XMAX-params->XMIN)/2.){
             dx+=params->XMAX-params->XMIN;
             }
             */
            double r2 = (dx*dx + dy*dy);//,0.5);
            double z=h2-r2;
            double dis=sqrt(r2);

            if(dis<=factor*h){
                double W=kernelW(factor,dis,h);
                delta=mass/rho[i]*W;
                vxb[ib]+=delta*vx[i];
                vyb[ib]+=delta*vy[i];
                Tb[ib]+=delta*Tf[i];
            }
        }
        //      if((s->yb[ib]-params->YMAX)<1.e-10){// uper layer
        //      s->Tb[ib]=0;
        // }
    }

    for (int i = 0; i < n; ++i) {

        fbx[i]=0;fby[i]=0;hb[i]=0;
        for (int ib = 0; ib < nb; ++ib) {

            int ib_is_top=0;
            if(fabs(yb[ib]-YMAX)<1.e-10){// uper layer
                ib_is_top=1;
            }

            double dx = xp[i]-xb[ib];

            /*if(dx>(params->XMAX-params->XMIN)/2.){
             dx-=params->XMAX-params->XMIN;
             }
             else if(dx<-(params->XMAX-params->XMIN)/2.){
             dx+=params->XMAX-params->XMIN;
             }
             */

            double dy = yp[i]-yb[ib];
            double dis=sqrt(dx*dx + dy*dy);
            if (dis<=factor*h) {
                double W=kernelW(factor,dis,h);
                double delta=mass*W;
                double buoy=g*beta*(Tb[ib]-Tf0);
                fbx[i]+=delta*(Vpx-vxb[ib])/dt-0;//+pforcex+visforcex;
                fby[i]+=delta*((Vpy-vyb[ib])/dt-buoy);//+pforcey+visforcey;
                if(ib_is_top==0)
                    hb[i]+=delta*(Tsb[ib]-Tb[ib])/dt;
            }
        }
    }
    /*
     // update position
     if(params->type==2){ //moving particle
     params->xc+=params->Vpx*params->dt;
     if(params->xc>params->XMAX){
     params->xc-=params->XMAX;
     }
     else if(params->xc<params->XMIN){
     params->xc+=params->XMAX;
     }

     params->yc+=params->Vpy*params->dt;

     double theta=0,dTheta=2*M_PI/params->Bnodes;

     for(int i=0;i<params->Bnodes;i++){
     theta+=dTheta;
     s->xb[i]=params->xc+(params->rad-0.5*params->hh)*cos(theta);
     s->yb[i]=params->yc+(params->rad-0.5*params->hh)*sin(theta);
     }
     }
     */
}

void sph_fluid::solid_movement()
{
    if( type==4){//only move the center
        xc+=Vpx*dt;
        if(xc>XMAX){
            xc-=XMAX;
        }
        else if(xc<XMIN){
            xc+=XMAX;
        }
        yc+=Vpy*dt;
    }
    else if(type==2){ //moving the center and bnodes
        xc+=Vpx*dt;
        if(xc>XMAX){
            xc-=XMAX;
        }
        else if(xc<XMIN){
            xc+=XMAX;
        }

        yc+=Vpy*dt;

        double theta=0,dTheta=2*M_PI/Bnodes;

        for(int i=0;i<Bnodes;i++){
            theta+=dTheta;
            xb[i]=xc+(rad-0.5*hh)*cos(theta);
            yb[i]=yc+(rad-0.5*hh)*sin(theta);
        }
    }
}

void sph_fluid::compute_time_derivatives()
{
    for (int i = 0; i < n; ++i) {
        if(solid[i]==0){ // fluid
            ax[i] = dpdx/rho[i]+fbx[i]/rho[i];
            ay[i] = fby[i]/rho[i]+beta*(Tf[i]-Tf0)*g;
            drho[i] = 0;
            dT[i]=hb[i]/rho[i];
        }
        else{
            ax[i] = fbx[i]/rho[i];
            ay[i] = fby[i]/rho[i];
            drho[i]=0;
            dT[i]=hb[i]/rho[i];
        }
    }
    double C=cspeed*max_vel();
    //    printf("max vel=%.5e \n",C);

    //    double C=(fabs(params->cspeed)>1.e-10 ? params->cspeed:0.00001);
    double Cs2=C*C;
    double Lx=XMAX-XMIN,Ly=YMAX-YMIN;
    // Now compute interaction forces

    for (int i = 0; i < n; ++i) {
        if(solid[i]==0 || solid[i]==5){// fluid point

            if(type==3 && solid[i]==5){//fixed non-ibm

            }
            else{

                double rhoi = rho[i];

                double Pi;
                if(PRESSURE==0){
                    Pi=Cs2*(rhoi);
                }
                else if(PRESSURE==1){
                    Pi=Cs2*(rhoi-rho0);//-params->rho0);
                }
                else if(PRESSURE==2){
                    Pi=Cs2*rho0*(pow(rhoi/rho0,7));
                }
                else if(PRESSURE==3){
                    Pi=Cs2*rho0*(pow(rhoi/rho0,7)-1);//-params->rho0);
                }

                for(int j=0;j<neighborList[i][MAX_LISTELEM-1];j++){
                    int jj=neighborList[i][j];
                    double dx = xp[i]-xp[jj];

                    /*
                     if(dx>Lx/2.){
                     dx=dx-Lx;
                     }
                     else if(dx<-Lx/2.){
                     dx=dx+Lx;
                     }
                     */

                    double dy = yp[i]-yp[jj];
                    /*if(dy>Ly/2.){
                     dy-=Ly;
                     }
                     else if(dy<-Ly/2.){
                     dy+=Ly;
                     }*/

                    double r2=dx*dx + dy*dy;
                    double dis = sqrt(r2);
                    double Tw_up=Tw_up,Tw_bot=Tw_bot;
                    double dTf;
                    double dvx,dvy,db_da,beta=1.5;
                    if(solid[jj]==0 || type==1 ){//||s->solid[jj]==5){
                        dvx = vx[i]-vx[jj];
                        dvy = vy[i]-vy[jj];
                        dTf=Tf[i]-Tf[jj];
                        if(solid[jj]==4){
                            dTf=0;// no flux
                        }

                    }
                    else if(solid[jj]==1){
                        db_da=(-xp[jj])/max(1.e-10,xp[i]);
                        dvx = vx[i]*min(beta,1+db_da);
                        dvy = vy[i]*min(beta,1+db_da);
                        dTf = (Tf[i]-0)*min(beta,1+db_da);
                    }
                    else if(solid[jj]==2){
                        db_da=(-XMAX+xp[jj])/max(1.e-10,XMAX-xp[i]);
                        dvx = (vx[i]-0)*min(beta,1+db_da);
                        dvy = (vy[i]-0)*min(beta,1+db_da);
                        dTf = (Tf[i]-0)*min(beta,1+db_da);
                    }
                    else if(solid[jj]==3){
                        db_da=(-yp[jj])/max(1.e-10,yp[i]);
                        dvx = vx[i]*min(beta,1+db_da);
                        dvy = vy[i]*min(beta,1+db_da);
                        dTf = 0;//(s->Tf[i]-Tw_bot)*min(beta,1+db_da); no flux
                    }
                    else if(solid[jj]==4){
                        db_da=(-YMAX+yp[jj])/max(1.e-10,YMAX-yp[i]);
                        dvx = (vx[i]-Utop)*min(beta,1+db_da);
                        dvy = (vy[i]-Vtop)*min(beta,1+db_da);
                        dTf = 0;//(s->Tf[i]-Tw_up)*min(beta,1+db_da); no flux
                    }
                    else if(solid[jj]==5){
                        db_da=db_da_for_cylinder(xp[i],yp[i],xp[jj],yp[jj],xc,yc,rad);
                        dvx = (vx[i]-Vpx)*min(beta,1+db_da);
                        dvy = (vy[i]-Vpy)*min(beta,1+db_da);
                        dTf = (Tf[i]-Ts)*min(beta,1+db_da);
                        //printf("i=%d jj=%d dvx=%e dTf=%e\n",i,jj,dvx,dTf);
                    }

                    double rhoj = rho[jj];
                    double rhos, Pj;
                    /*
                     if(s->solid[jj]==0||((params->type==1||params->type==2) &&(s->solid[jj]==5))){
                     rhos=s->rho[jj];
                     }
                     else{
                     rhos=params->rho0; // slightly increase solid node pressure
                     }
                     */
                    rhos=rhoj;

                    if(PRESSURE==0){
                        Pj=Cs2*(rhos);
                    }
                    else if(PRESSURE==1){
                        Pj=Cs2*(rhos-rho0);//-params->rho0);
                    }
                    else if(PRESSURE==2){
                        Pj=Cs2*rho0*(pow(rhos/rho0,7));
                    }
                    else if(PRESSURE==3){
                        Pj=Cs2*rho0*(pow(rhos/rho0,7)-1.);//-params->rho0);
                    }

                    double Wx,Wy;

                    gradientKernelW(factor,dx,dy,dis,h,&Wx,&Wy);

                    double rho_ij = mass*((dvx)*Wx+(dvy)*Wy);
                    drho[i] += rho_ij;

                    double pres=(Pi/(rhoi*rhoi)+Pj/(rhoj*rhoj));

                    if(TENSILE==1){ // add additional normal stress
                        //            double Rij=0.3*(Pi)/(rhoi*rhoi);

                        double FEij=kernelW(factor,dis,h)/Wh0;

                        //            double art_pres=Rij*FEij*FEij*FEij*FEij;

                        pres=pres*(1+0.2*FEij*FEij*FEij*FEij);
                    }

                    double visf=2*mu/((rhoi*rhoj)*(r2+0.01*h*h))*(dx*Wx+dy*Wy);
                    //double buoy=params->beta*params->g*(s->Tf[i]-params->Tf);

                    ax[i] += mass*(-pres*Wx + visf*dvx);
                    ay[i] += mass*(-pres*Wy + visf*dvy);// + buoy*kernelW(params->factor,dis,params->h));

                    double Tt=alpha* (rhoi+rhoj)/((rhoi*rhoj)*(r2+0.01*h*h))*(dx*Wx+dy*Wy);
                    dT[i]+=mass*Tt*dTf;
                }
            }
        }
    }
}

double sph_fluid::db_da_for_cylinder(double xi,double yi, double xj,double yj, double xc, double yc, double rad)
{
    double di, dj, x1, y1, x1P, x1M,x2,y2;
    double m1, m2,rad2=rad*rad;
    if(fabs(xi - xc) >1.e-20){
        m1 = (yi - yc) / (xi - xc);
        double tmp=rad2/(1+m1*m1);
        x1P=tmp+xc;
        x1M=-tmp+xc;
        // which one we need
        if(xc < xi){
            if(xc <= x1P && x1P <= xc)
                x1 = x1P;
            else
                x1 = x1M;
        } else {
            if(xi <= x1P && x1P <= xc)
                x1 = x1P;
            else
                x1 = x1M;
        }
        y1 = m1*(x1 - xc) + yc;
    } else {
        x1 = xi;
        // which intersection
        if(yc < yi)
            y1 = yc + rad;
        else
            y1 = yc -rad;
    }
    // find point is the minimum (x2, y2).
    if (fabs(m1)>1.e-10){//
        m2 = -1.0 / m1;
        x2 =(yj - y1 + m2*x1-m1*xj) / (m2 - m1);
        y2 = m1*(x2 -xj) + yj;

        dj = sqrt((xj - x2)*(xj - x2) + (yj - y2)*(yj - y2));
        di = sqrt((xi - x1)*(xi - x1) + (yi - y1)*(yi - y1));

        return (dj / di);
    } else {
        dj = fabs(yj - y1);
        di = fabs(yi - y1);
        return (dj / di);
    }
}

void sph_fluid::movement_step()
{
    for (int i = 0; i < n; ++i) {
        if(solid[i]==0){
            vx[i]  = vx[i] + ax[i] * dt;
            vy[i]  = vy[i] + ay[i] * dt;
            rho[i] = rho[i] + drho[i] * dt;
            Tf[i] = Tf[i] + dT[i] * dt;
        }
    }

    for (int i = 0; i < n; ++i) {
        if(solid[i]==0){
            xp[i]  += vx[i] * dt;
            if(xp[i]>XMAX || xp[i]<XMIN) {
                xp[i]=xp[i]-vx[i]*dt;
                vx[i]=-vx[i]; // reflect
            }

            yp[i]  += vy[i] * dt;

            if(yp[i]>YMAX || yp[i]<YMIN) {
                yp[i]=yp[i]-vy[i]*dt;
                vy[i]=-vy[i];
            }

            double dis=sqrt((xp[i]-xc)*(xp[i]-xc)+
                            (yp[i]-yc)*(yp[i]-yc));
            if(dis<rad){
                xp[i]=xp[i]-vx[i]*dt;
                yp[i]=yp[i]-vy[i]*dt;
                vx[i]=-vx[i];
                vy[i]=-vy[i];
            }

        }

	

        if(solid[i]==5 && type==4){
            xp[i]  += Vpx * dt;
            if(xp[i]>XMAX) xp[i]=xp[i]-XMAX;
            if(xp[i]<XMIN) xp[i]=xp[i]+XMAX;
            yp[i]  += Vpy * dt;
            if(yp[i]>YMAX) yp[i]=yp[i]-YMAX;
            if(yp[i]<YMIN) yp[i]=yp[i]+YMAX;
            // moving at constant speed
        }
    }
}

void sph_fluid::compute_DragHeat()
{
    BodyFx=0;
    BodyFy=0;
    BodyNu=0;
    double dV=2*M_PI*(rad)/nb*hh;

    for (int ib = 0; ib < nb; ++ib) {
        BodyFx+=(Vpx-vxb[ib])/dt;
        BodyFy+=(Vpy-vyb[ib])/dt;
        BodyNu+=(Tsb[ib]-Tb[ib])/dt;
    }
    if(type==2){
        BodyFx=BodyFx*dV/(.5*Vpx*Vpx*rad*2);
    }

    BodyNu/=(Ts-0);

}

void sph_fluid::normalize_mass()
{
    double rho2s = 0;
    double rhos  = 0;
    for (int i = 0; i < n; ++i) {
        rho[i]=rho0;
        rhoh[i]=rho0;
    }
    mass=hh*hh*rho0;

}

double** sph_fluid:: make2dmem(int arraySizeX, int arraySizeY)
{
    int i,j;

    double** theArray;
    theArray = (double**) malloc(arraySizeX*sizeof(double*));
    for (i = 0; i < arraySizeX; i++)
        theArray[i] = (double*) malloc(arraySizeY*sizeof(double));
    // always clear it
    for(i=0;i<arraySizeX;i++){
        for(j=0;j<arraySizeY;j++){
            theArray[i][j]=0.;
        }
    }
    return theArray;
}

int** sph_fluid::make2dmemInt(int arraySizeX, int arraySizeY)
{
    int i,j;

    int** theArray;
    theArray = (int**) malloc(arraySizeX*sizeof(int*));
    for (i = 0; i < arraySizeX; i++)
        theArray[i] = (int*) malloc(arraySizeY*sizeof(int));
    // always clear it
    for(i=0;i<arraySizeX;i++){
        for(j=0;j<arraySizeY;j++){
            theArray[i][j]=0;
        }
    }
    return theArray;
}

void sph_fluid::alloc_state(int n0, int nb0, int MAX_LISTELEM)
{
    n   =  n0;
    nb=nb0;
    Tsb=(double*) calloc(nb,sizeof(double));

    rho =  (double*) calloc(  n, sizeof(double));
    xp =    (double*) calloc(n, sizeof(double));
    yp =    (double*) calloc(n, sizeof(double));
    rhoh =   (double*) calloc(n, sizeof(double));

    vxh =   (double*) calloc(n, sizeof(double));
    vyh =   (double*) calloc(n, sizeof(double));
    vx =    (double*) calloc(n, sizeof(double));
    vy =    (double*) calloc(n, sizeof(double));

    drho =    (double*) calloc(n, sizeof(double));

    Tf =    (double*) calloc(n, sizeof(double));
    Tfh =    (double*) calloc(n, sizeof(double));
    dT =    (double*) calloc(n, sizeof(double));

    ax =    (double*) calloc(n, sizeof(double));
    ay =    (double*) calloc(n, sizeof(double));
    fbx =    (double*) calloc(n, sizeof(double));
    fby =    (double*) calloc(n, sizeof(double));
    hb =    (double*) calloc(n, sizeof(double));

    solid =  (int*) calloc(n, sizeof(int));

    xb =    (double*) calloc(nb, sizeof(double));
    yb =    (double*) calloc(nb, sizeof(double));
    vxb =    (double*) calloc(nb, sizeof(double));
    vyb =    (double*) calloc(nb, sizeof(double));

    Tb =    (double*) calloc(nb, sizeof(double));

    neighborList=make2dmemInt(n,MAX_LISTELEM);
    neighborList=(int **) calloc(n,sizeof(int *));
    for(int i=0;i<n;i++){
        neighborList[i]=(int *)calloc(MAX_LISTELEM,sizeof(int));
    }

}


void sph_fluid::place_particles()
{
    double Rb=rad;
    int bcount_cyl=(int)(M_PI*Rb/hh);

    // wall nodes
    int bcount_wallx=(int)((XMAX-XMIN)/hh);
    int bcount_wally=(int)((YMAX-YMIN)/hh);
    Bnodes=bcount_cyl+2*bcount_wallx+2*bcount_wally;      //question
    double dTheta=(2*M_PI)/bcount_cyl;
    double theta=0;
    double epson=0;//Rb/2.;//0.01;
    double XC=(XMAX-XMIN)/2.;
    double YC=(YMAX-YMIN)/2.+epson;
    xc=XC;                              //update base class param cylinder position
    yc=YC;                              //update base class param cylinder position
    // Count mesh points that fall in indicated region.
    int count = 0;
    int mx=(XMAX-XMIN)/hh+2;
    int cy=0;

    if(PLACEMENT==1)
    {// hexogan

        int NB=(factor==2? 5:9);// ceil(param->factor*1.5);
        for (double y = -NB*hh/2.; y <=YMAX+(NB+.1)*hh/2; y += hh){
            if(cy%2==0){
                for (double x = -NB*hh/2; x <=XMAX+(NB+.1)*hh/2; x += hh){
                    count += 1;
                }
            }
            else{
                for (double x = -NB*hh/2+hh/2; x <=XMAX+(NB+.1)*hh/2; x += hh){
                    count += 1;
                }
            }

            cy++;
            }
        printf("count of particles %d\n",count);
        // Populate the particle data structure
        alloc_state(count,Bnodes,MAX_LISTELEM);

        for(int i=0;i<bcount_cyl;i++){
            theta+=dTheta;
            xb[i]=XC+(Rb-0.5*hh)*cos(theta); //retract radius
            yb[i]=YC+(Rb-0.5*hh)*sin(theta);
            Tsb[i]=Ts;
        }

        for(int i=bcount_cyl;i<bcount_cyl+bcount_wallx;i++){
            xb[i]=XMIN+(i-bcount_cyl)*(XMAX-XMIN)/bcount_wallx;
            yb[i]=YMIN;
            Tsb[i]=0;

        }
        for(int i=bcount_cyl+bcount_wallx;i<bcount_cyl+bcount_wallx+bcount_wally;i++){
            xb[i]=XMAX;
            yb[i]=YMIN+(i-bcount_cyl-bcount_wallx)*(YMAX-YMIN)/bcount_wally;
            Tsb[i]=0;
        }
        for(int i=bcount_cyl+bcount_wallx+bcount_wally;i<bcount_cyl+2*bcount_wallx+bcount_wally;i++){
            xb[i]=XMAX-(i-bcount_cyl-bcount_wallx-bcount_wally)*(XMAX-XMIN)/bcount_wallx;
            yb[i]=YMAX;
            Tsb[i]=0;
        }
        for(int i=bcount_cyl+2*bcount_wallx+bcount_wally;i<bcount_cyl+2*bcount_wallx+2*bcount_wally;i++){
            xb[i]=XMIN;
            yb[i]=YMAX-(i-bcount_cyl-2*bcount_wallx-bcount_wally)*(YMAX-YMIN)/bcount_wally;
            Tsb[i]=0;
        }
        double dis;
        int p = 0;
        cy=0;

        for (double y = -NB*hh/2.; y <=YMAX+(NB+.1)*hh/2; y += hh){
            if(cy%2==0){
                for (double x = -NB*hh/2; x <=XMAX+(NB+.1)*hh/2; x += hh){


                    //for (double y = 0; y <=YMAX-0.001*hh; y += hh){
                    //if(cy%2==0){
                    //  for (double x = 0; x <=XMAX-0.001*hh; x += hh){

                    xp[p] = x;
                    yp[p] = y;
                    solid[p]=0; //fluid

                    if(yp[p]>YMAX){ //upper
                        // printf("x=%lf y=%lf\n",s->xp[p],s->yp[p]);
                        solid[p]=4;
                    }
                    else if(yp[p]<YMIN){ //bottom
                        solid[p]=3;
                    }
                    else if(xp[p]<XMIN){ //left side
                        solid[p]=1;
                    }
                    else if(xp[p]>XMAX){ //right side
                        solid[p]=2;
                    }

                    dis=sqrt((xp[p]-XC)*(xp[p]-XC)+
                             (yp[p]-YC)*(yp[p]-YC));
                    if(dis<=Rb){
                        if(type==3 || type==4){
                            solid[p]=5;
                        }
                    }
                    vx[p] = 0;
                    vy[p] = 0;
                    vxh[p] = 0;
                    vyh[p] = 0;

                    Tf[p] = Tf0;
                    Tfh[p] = Tf0;

                    ax[p] = 0;
                    ay[p] = 0;
                    fbx[p] = 0;
                    fby[p] = 0;
                    hb[p]=0;
                    p++;
                }
            }
        else{
            for (double x = -NB*hh/2+hh/2; x <=XMAX+(NB+.1)*hh/2; x += hh){
                //for (double x = hh/2; x <=XMAX; x += hh){
                xp[p] = x;
                yp[p] = y;
                solid[p]=0; //fluid

                if(yp[p]>YMAX){ //upper
                    // printf("x=%lf y=%lf\n",s->xp[p],s->yp[p]);
                    solid[p]=4;
                }
                else if(yp[p]<YMIN){ //bottom
                    solid[p]=3;
                }
                else if(xp[p]<XMIN){ //left side
                    solid[p]=1;
                }
                else if(xp[p]>XMAX){ //right side
                    solid[p]=2;
                }

                dis=sqrt((xp[p]-XC)*(xp[p]-XC)+
                         (yp[p]-YC)*(yp[p]-YC));
                if(dis<=Rb){
                    if(type==3 || type==4){
                        solid[p]=5;  // inside cylinder
                    }
                }

                vx[p] = 0;
                vy[p] = 0;
                vxh[p] = 0;
                vyh[p] = 0;
                Tf[p] = Tf0;
                Tfh[p] = Tf0;
                ax[p] = 0;
                ay[p] = 0;
                fbx[p] = 0;
                fby[p] = 0;
                hb[p]=0;
                p++;
            }

        }

        cy++;
    }

}
else{// regular
    int NB=(factor==2? 5:9);// ceil(param->factor*1.5);
    for (double x = -NB*hh/2; x <=XMAX+(NB+.1)*hh/2; x += hh)
        for (double y = -NB*hh/2.; y <=YMAX+(NB+.1)*hh/2; y += hh)
            count += 1;

    printf("count of particles %d\n",count);

    // Populate the particle data structure
    alloc_state(count,Bnodes,MAX_LISTELEM);

    for(int i=0;i<bcount_cyl;i++){
        theta+=dTheta;
        xb[i]=XC+(Rb-0.5*hh)*cos(theta); //retract radius
        yb[i]=YC+(Rb-0.5*hh)*sin(theta);
        Tsb[i]=Ts;
    }

    for(int i=bcount_cyl;i<bcount_cyl+bcount_wallx;i++){
        xb[i]=XMIN+(i-bcount_cyl)*(XMAX-XMIN)/bcount_wallx;
        yb[i]=YMIN;
        Tsb[i]=0;
    }
    for(int i=bcount_cyl+bcount_wallx;i<bcount_cyl+bcount_wallx+bcount_wally;i++){
        xb[i]=XMAX;
        yb[i]=YMIN+(i-bcount_cyl-bcount_wallx)*(YMAX-YMIN)/bcount_wally;
        Tsb[i]=0;
    }
    for(int i=bcount_cyl+bcount_wallx+bcount_wally;i<bcount_cyl+2*bcount_wallx+bcount_wally;i++){
        xb[i]=XMAX-(i-bcount_cyl-bcount_wallx-bcount_wally)*(XMAX-XMIN)/bcount_wallx;
        yb[i]=YMAX;
        Tsb[i]=0;
    }
    for(int i=bcount_cyl+2*bcount_wallx+bcount_wally;i<bcount_cyl+2*bcount_wallx+2*bcount_wally;i++){
        xb[i]=XMIN;
        yb[i]=YMAX-(i-bcount_cyl-2*bcount_wallx-bcount_wally)*(YMAX-YMIN)/bcount_wally;
        Tsb[i]=0;
    }


    double dis;
    int p = 0;
    //      for (double x = hh/2.; x <XMAX; x += hh) {
    for (double x = -NB*hh/2; x <=XMAX+(NB+.1)*hh/2; x += hh){

        for (double y = -NB*hh/2.; y <=(NB+.1)*hh/2+YMAX; y += hh) {
            xp[p] = x;
            yp[p] = y;
            vx[p] = 0;
            vy[p] = 0;
            vxh[p] = 0;
            vyh[p] = 0;
            Tf[p] = Tf0;
            Tfh[p] = Tf0;
            ax[p] = 0;
            ay[p] = 0;
            fbx[p] = 0;
            fby[p] = 0;
            hb[p]=0;

            solid[p]=0; //fluid

            if(yp[p]>YMAX){ //upper
                // printf("x=%lf y=%lf\n",s->xp[p],s->yp[p]);
                solid[p]=4;
            }
            else if(yp[p]<YMIN){ //bottom
                solid[p]=3;
            }
            else if(xp[p]<XMIN){ //left side
                solid[p]=1;
            }
            else if(xp[p]>XMAX){ //right side
                solid[p]=2;
            }
            dis=sqrt((xp[p]-XC)*(xp[p]-XC)+
                     (yp[p]-YC)*(yp[p]-YC));
            if(dis<=Rb){
                if(type==3 || type==4){
                    solid[p]=5;  // inside cylinder
                    //printf("p=%d \n",p);
                }
            }
            ++p;
        }
    }

}
}

void sph_fluid::print_frame(FILE *fp_solid)
{
    int j;
    
    fprintf(fp_solid,"# vtk DataFile Version 2.0\n");
    fprintf(fp_solid,"Particle Tracking\n");
    fprintf(fp_solid,"ASCII\n\n");
    
    fprintf(fp_solid,"DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp_solid,"POINTS %d double\n",n);
    
    for(j=0;j<n;j++){
      fprintf(fp_solid,"%.6e  %.6e 0\n",xp[j],yp[j]);
    }
    
    fprintf(fp_solid,"\nPOINT_DATA %d\n",n);
    
    fprintf(fp_solid,"SCALARS diameter double 1\n");
    fprintf(fp_solid,"LOOKUP_TABLE DEFAULT\n");
    for(j=0;j<n;j++){
      if(solid[j]==0){
        fprintf(fp_solid,"%.4e\n",hh);//det_radius(j));
      }
      else{
        fprintf(fp_solid,"%.4e\n",1.1*hh);//det_radius(j));
      }
    }
    
    fprintf(fp_solid,"SCALARS Temperature double 1\n");
    fprintf(fp_solid,"LOOKUP_TABLE DEFAULT\n");
    for(j=0;j<n;j++){
      double Tft=0;
      if(type==3 && solid[j]==5){
        Tft=Ts;
      }
      if(fabs(Tf[j])>1.e-6){
        Tft=Tf[j];
      }
      fprintf(fp_solid,"%.4e\n",Tft);
      
    }
//  fprintf(fp_solid,"\nPOINT_DATA %d\n",state->n);
    fprintf(fp_solid,"VECTORS velocity double\n");
    double vxt,vyt;
    for(j=0;j<n;j++){
      if(fabs(vx[j])<1.e-10){
        vxt=0;
      }
      else{
        vxt=vx[j];
      }
      if(fabs(vy[j])<1.e-10){
        vyt=0;
      }
      else{
        vyt=vy[j];
      }
      
      fprintf(fp_solid,"%.4e %.4e 0\n",vxt,vyt);//state->vx[j],state->vy[j]);//det_radius(j));
    }


    fprintf(fp_solid,"SCALARS velo double\n");
    fprintf(fp_solid,"LOOKUP_TABLE DEFAULT\n");
    double vxtv,vytv;
    for(j=0;j<n;j++){
      if(fabs(vx[j])<1.e-10){
        vxtv=0;
      }
      else{
        vxtv=vx[j];
      }
      if(fabs(vy[j])<1.e-10){
        vytv=0;
      }
      else{
        vytv=vy[j];
      }
      fprintf(fp_solid,"%.4e\n",sqrt(vxtv*vxtv+vytv*vytv));
      //fprintf(fp_solid,"%.4e %.4e 0\n",vxt,vyt);//state->vx[j],state->vy[j]);//det_radius(j));
    }
    
}


#endif // PARTICLE_H


