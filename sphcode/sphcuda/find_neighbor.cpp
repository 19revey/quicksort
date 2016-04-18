// gcc mysphv2.c -std=gnu99 -Wall -g -lm
//SPH method to simulate fluid motion
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define max(x,y)  (x>y? x:y)
void Preprocesses(int d, int n,double **PointSet, double **OrderedSet, int *BMap, int **FMap);
void Quick_Sort(int n, double *PointSet,int *TempMap);
void Bubble_Sort(int n, double *PointSet,int *TempMap);
int Closest(int d, int n, double *P,double Epsilon, double **OrderedSet,
	    int *BMap,int **FMap,int *resL,int MAX_LISTELE);
int  BinarySearch(int n, double *OrderedSet,double v);
extern double** make2dmem(int arraySizeX, int arraySizeY);
extern int** make2dmemInt(int arraySizeX, int arraySizeY);

void find_neighbor_spatial(int n,double *xp,double *yp,double dis,int **neighborList,int MAX_LISTELEM);
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Find the neighbors (particles within a distance 'dis') of each particle among n particles
//-----------------------------------------------------------------------------------------------------
// INPUT
//                  n:  number of the particles
//                *xp:  position x coordinates
//                *yp:  position y coordinates
//                dis:  distance threshold for becomming a neighbor
//        MAX_LISTLEM:  allowed maximum list length 
//
//  OUTPUT
//  nighborList[i][j]:  store the indices of particle i's neighboring particles.
//                      neighborList[i][MAX_LISTELEM-1], the last element in the list, stores the number
//                      of neighboring particles.
////////////////////////////////////////////////////////////////////////////////////////////////////////
void sph_fluid::find_neighbor_spatial(int n,double *xp,double *yp,double dis,int **neighborList, int MAX_LISTELEM)
{
    int d=2; // 2 dimension
    double **PointSet=make2dmem(d,n);
    double **OrderedSet=make2dmem(d,n);
    double *P=(double*)calloc(d,sizeof(double));
    int *BMap=(int*)calloc(n,sizeof(int));
    int **FMap=make2dmemInt(d,n);

    memcpy(PointSet[0],xp,n*sizeof(double));
    memcpy(PointSet[1],yp,n*sizeof(double));
    //for(int i=0;i<n;i++){
    //PointSet[0][i]=xp[i];
    //PointSet[1][i]=yp[i];
    //}
    Preprocesses(2,n,PointSet,OrderedSet,BMap,FMap);    

    for(int i=0;i<n;i++){
      P[0]=PointSet[0][i];
      P[1]=PointSet[1][i];
      int id=Closest(d,n,P,dis,OrderedSet,BMap,FMap,neighborList[i],MAX_LISTELEM); 
      neighborList[i][MAX_LISTELEM-1]=id; // last element used to store number of neighbors
      //  printf("in spatial i=%d list=%d\n",i,id);
    }
    
    for(int i=0;i<d;i++){
      free(PointSet[i]);
      free(OrderedSet[i]);
      free(FMap[i]);
    }
    
    ////////////////
    /*
    for(int i=40000; i<40001; i++)
    {
        printf("%.4d,      %.4d,   %.4d,   %.4d,   %.4d,      %.4d\n",i, neighborList[i][0],neighborList[i][1],neighborList[i][2],neighborList[i][3],neighborList[i][MAX_LISTELEM-1]);
    }*/
    //////////////////
    
    
    
    free(PointSet);
    free(OrderedSet);
    free(FMap);
    free(BMap);
    free(P);
}


void Preprocesses(int d,int n,double **PointSet, double **OrderedSet, int *BMap, int **FMap)
{
    int *TempMap=(int*)calloc(n,sizeof(int));
    for(int i=0;i<d;i++){
      Quick_Sort(n,PointSet[i],TempMap);
      
      for(int j=0;j<n;j++){
        OrderedSet[i][j]=PointSet[i][TempMap[j]];
        if(i==0){
          BMap[j]=TempMap[j];
        }
        FMap[i][TempMap[j]]=j;
      }
    }
    free(TempMap);
}

// bubble sort
void Bubble_Sort(int n, double *PointSet,int *TempMap)
{
  for(int i=0;i<n;i++){
      TempMap[i]=i;
  }
  
  for(int i=0;i<n;i++){
      for(int j=0;j<n-i-1;j++){
          if(PointSet[TempMap[j]]>PointSet[TempMap[j+1]]){
              int t=TempMap[j+1];
              TempMap[j+1]=TempMap[j];
              TempMap[j]=t;
          }
      }
  }
}

int Closest(int d, int n, double *P,double Epsilon, double **OrderedSet,
	    int *BMap,int **FMap,int *resL,int MAX_LISTELEM)
{
  int Bottom=BinarySearch(n,OrderedSet[0],P[0]-Epsilon);//+1;
  int Top=BinarySearch(n,OrderedSet[0],P[0]+Epsilon);
  int List[n];
  int i,j;
  int ListElem=0;
  double Dis2=Epsilon*Epsilon;
  
  for(i=Bottom;i<=Top;i++){
    List[ListElem++]=BMap[i];
  }
  
  for(i=1;i<d;i++){
    Bottom=BinarySearch(n,OrderedSet[i],P[i]-Epsilon);//+1;
    Top=BinarySearch(n,OrderedSet[i],P[i]+Epsilon);  
    int m=ListElem;
    ListElem=0;
    
    for(j=0;j<m;j++){      
      if(FMap[i][List[j]]<=Top && FMap[i][List[j]]>=Bottom){
        List[ListElem++]=List[j];
      }
    }
    if(ListElem>MAX_LISTELEM-1){
      printf("There are %d neighbours. The pre-allocated MAX_LISTELEM (%d) is too small.\n",ListElem,MAX_LISTELEM);
     // exit(0);
    }
  }
  
  double max=100000000.0;
  int count=0;
  for(i=0;i<ListElem;i++){
    double t=0;
    for(j=0;j<d;j++){
      double dD=(P[j]-OrderedSet[j][FMap[j][List[i]]]);
      t+=dD*dD;
    }   
    if(t<Dis2 && t>1e-50){
      resL[count]=List[i];
      count++;
    }
  }
    if(count>MAX_LISTELEM-1){
        printf("There are %d real neighbours. The pre-allocated MAX_LISTELEM (%d) is too small.\n",ListElem,MAX_LISTELEM);
         exit(0);
    }
  return count;
}

int  BinarySearch(int n,double *OrderedSet,double v)
{
    int Bottom=0;
    int Top=n;
    while(Top>Bottom+1){
      int Center=(Top+Bottom)/2;
      if(v<OrderedSet[Center]){
        Top=Center;
      }
      else{
        Bottom=Center;
      }
    }
    return Bottom;
}  
// allocate memory and initilize it  
double** make2dmem1(int arraySizeX, int arraySizeY)
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

int** make2dmemInt1(int arraySizeX, int arraySizeY)
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


typedef struct _amplitude_index{
// Struct to store and order the values of the amplitudes preserving the index in the original array
    double amplitude;
    int index;
} t_amplitude_index;
 
int compare_structs (const void *a, const void *b);

typedef struct _value_index{
    double value;
    int index;
} t_value_index;

void Quick_Sort(int n, double *PointSet,int *TempMap)
{
    t_value_index *array_pt;
    array_pt=(t_value_index *)malloc(sizeof(t_value_index)*n);
  
    for(int i = 0; i< n;i++){
      array_pt[i].value = PointSet[i];
      array_pt[i].index = i;
    }
    
    qsort(array_pt, n, sizeof(array_pt[0]), compare_structs);
    for(int i=0;i<n;i++){
      TempMap[i]=array_pt[n-i-1].index;
    }
    free(array_pt);
}

int compare_structs(const void *a, const void *b)
{    
    t_value_index *struct_a = (t_value_index *) a;
    t_value_index *struct_b = (t_value_index *) b;
    
    if (struct_a->value < struct_b->value) return 1;
    else if (struct_a->value == struct_b->value) return 0;
    else return -1;  
}
