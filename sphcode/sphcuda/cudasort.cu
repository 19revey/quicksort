

#ifndef CUDA_H
#define CUDA_H



#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include <algorithm>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include "helper_math.h"
#include "math_constants.h"

#define USE_TEX 0

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif
#include "vector_types.h"
typedef unsigned int uint;


extern "C"
{

void generate_random_points(float *points, int numofpoints)
{
    // sequentially generate some random 2D points in the unit square
    std::cout << "generating points\n" << std::endl;
    
    for(int i = 0; i < numofpoints; ++i)
    {
        //srand (time(NULL));
        points[i*4] = float(rand()) / RAND_MAX;
        points[i*4+1] = float(rand()) / RAND_MAX;
        points[i*4+2] = float(rand()) / RAND_MAX;
        points[i*4+3] = 0;
    }
}

void allocateArray(void **devPtr, size_t size)
{
    checkCudaErrors(cudaMalloc(devPtr, size));
}
//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


// calculate grid hash value for each particle
// calculate position in uniform grid
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}
__device__ int3 calcGridPos(float3 p, float3 worldOrigin, float3 cellSize)
{
    int3 gridPos;
    gridPos.x = floor((p.x -  worldOrigin.x) /  cellSize.x);
    gridPos.y = floor((p.y -  worldOrigin.y) /  cellSize.y);
    gridPos.z = floor((p.z -  worldOrigin.z) /  cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos, uint3 gridSize)
{
    gridPos.x = gridPos.x & (gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (gridSize.y-1);
    gridPos.z = gridPos.z & (gridSize.z-1);
    return gridPos.z*gridSize.y*gridSize.x + gridPos.y*gridSize.x + gridPos.x;
}

__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles,
               float3 worldOrigin,
               uint3 gridSize,
               float3 cellSize)
{
    uint index = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if (index >= numParticles) return;
    
    volatile float4 p = pos[index];
    
    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z),worldOrigin, cellSize);
    uint hash = calcGridHash(gridPos, gridSize);
    
    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}
void calcHash(uint  *gridParticleHash,
              uint  *gridParticleIndex,
              float *pos,
              int    numParticles,
              float3 worldOrigin,
              uint3 gridSize,
              float3 cellSize)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
    
    // execute the kernel
    calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                           gridParticleIndex,
                                           (float4 *) pos,numParticles,worldOrigin,gridSize,cellSize);
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


// sorting particles by hash
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//

void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
{
    thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                        thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                        thrust::device_ptr<uint>(dGridParticleIndex));
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                 // float4 *sortedVel,        // output: sorted velocities
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float4 *oldPos,           // input: sorted position array
                                 // float4 *oldVel,           // input: sorted velocity array
                                  uint    numParticles)
{
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    uint hash;
    
    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];
        
        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;
        
        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }
    
    __syncthreads();
    
    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell
        
        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;
            
            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }
        
        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }
        
        // Now use the sorted index to reorder the pos and vel data
        uint sortedIndex = gridParticleIndex[index];
        float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
       // float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh
        
        sortedPos[index] = pos;
       // sortedVel[index] = vel;
    }
    
    
}

void reorderDataAndFindCellStart(uint  *cellStart,
                                 uint  *cellEnd,
                                 float *sortedPos,
                                 //float *sortedVel,
                                 uint  *gridParticleHash,
                                 uint  *gridParticleIndex,
                                 float *oldPos,
                                 //float *oldVel,
                                 uint   numParticles,
                                 uint   numCells)
{
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 256, numBlocks, numThreads);
    
    // set all cells to empty
    checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));
    
#if USE_TEX
    checkCudaErrors(cudaBindTexture(0, oldPosTex, oldPos, numParticles*sizeof(float4)));
    //checkCudaErrors(cudaBindTexture(0, oldVelTex, oldVel, numParticles*sizeof(float4)));
#endif
    
    uint smemSize = sizeof(uint)*(numThreads+1);
    reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
                                                                       cellStart,
                                                                       cellEnd,
                                                                       (float4 *) sortedPos,
                                                                       //(float4 *) sortedVel,
                                                                       gridParticleHash,
                                                                       gridParticleIndex,
                                                                       (float4 *) oldPos,
                                                                       //(float4 *) oldVel,
                                                                       numParticles);
    getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");
    
#if USE_TEX
    checkCudaErrors(cudaUnbindTexture(oldPosTex));
    //checkCudaErrors(cudaUnbindTexture(oldVelTex));
#endif
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// finde neighbour cell particles
//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
__device__
int collideSpheres(float3 posA, float3 posB,
                      float radiusA, float radiusB)
{
    // calculate relative position
    float3 relPos = posB - posA;
    
    float dist = length(relPos);
    float collideDist = radiusA + radiusB;
    
    //float3 force = make_float3(0.0f);
    
    if (dist < collideDist)
    {
        return 1;
    }
    else
        return 0;
    
}


/*
__device__
void collideCell(int3    gridPos,
                   uint    index,
                   float3  pos,
                   //float3  vel,
                   float4 *oldPos,
                   //float4 *oldVel,
                   uint   *cellStart,
                   uint   *cellEnd,
                   float particleRadius,
                   float3 worldOrigin,
                   uint3 gridSize,
                   float3 cellSize,
                   uint *d_neighborList)
{
    uint gridHash = calcGridHash(gridPos, gridSize);
    
    // get start of bucket for this cell
    uint startIndex = FETCH(cellStart, gridHash);
    
    //float3 force = make_float3(0.0f);
    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = FETCH(cellEnd, gridHash);
        
        uint a=0;
        for (uint j=startIndex; j<endIndex; j++)
        {

            if (j != index)                // check not colliding with self
            {
                float3 pos2 = make_float3(FETCH(oldPos, j));
                //float3 vel2 = make_float3(FETCH(oldVel, j));
                
                // collide two spheres
                if(collideSpheres(pos, pos2, particleRadius,particleRadius))
                {

                        d_neighborList[index*10+a]=d_neighborList[index*10+a]+j;
                        a=a+1;

                }
            }
        }
    }
    
}*/


__global__
void collideD(//float4 *newVel,               // output: new velocity
              float4 *oldPos,               // input: sorted positions
              //float4 *oldVel,               // input: sorted velocities
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint   numParticles,
              uint   numCells,
              float particleRadius,
              float3 worldOrigin,
              uint3 gridSize,
              float3 cellSize,
              uint *d_neighborList,
              uint MAX_LISTELEM)
{
    uint index = (blockIdx.x*blockDim.x) + threadIdx.x;
    
    if (index >= numParticles) return;
    
    // read particle data from sorted arrays
    float3 pos = make_float3(FETCH(oldPos, index));
    //float3 vel = make_float3(FETCH(oldVel, index));
    
    // get address in grid
    int3 gridPos = calcGridPos(pos, worldOrigin, cellSize);
    
    // examine neighbouring cells
    //float3 force = make_float3(0.0f);

    uint a=0;
    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {

                int3 neighbourPos = gridPos + make_int3(x, y, z);
               // collideCell(neighbourPos, index, pos, oldPos, cellStart, cellEnd, particleRadius,worldOrigin,gridSize,cellSize,d_neighborList);
                uint gridHash = calcGridHash(neighbourPos, gridSize);
                
                // get start of bucket for this cell
                uint startIndex = FETCH(cellStart, gridHash);
                
                //float3 force = make_float3(0.0f);
                if (startIndex != 0xffffffff)          // cell is not empty
                {
                    // iterate over particles in this cell
                    uint endIndex = FETCH(cellEnd, gridHash);
                    
                    
                    for (uint j=startIndex; j<endIndex; j++)
                    {
                        
                        if (j != index)                // check not colliding with self
                        {
                            float3 pos2 = make_float3(FETCH(oldPos, j));
                            //float3 vel2 = make_float3(FETCH(oldVel, j));
                            
                            // collide two spheres
                            if(collideSpheres(pos, pos2, particleRadius,particleRadius))
                            {
                                if (a>MAX_LISTELEM) {
                                    return;
                                }
                                //uint originalIndex = gridParticleIndex[index];
                                d_neighborList[index*MAX_LISTELEM+a]=gridParticleIndex[j];
                                a=a+1;
                                
                                
                            }
                        }
                    }
                }


            }
        }
    }

    d_neighborList[index*(MAX_LISTELEM)+MAX_LISTELEM-1]=a;
    
    // collide with cursor sphere
    //force += collideSpheres(pos, params.colliderPos, vel, make_float3(0.0f, 0.0f, 0.0f), params.particleRadius, params.colliderRadius, 0.0f);
    
    // write new velocity back to original unsorted location
//uint originalIndex = gridParticleIndex[index];
    //newVel[originalIndex] = make_float4(vel + force, 0.0f);
}

void collide(//float *newVel,
             float *sortedPos,
             //float *sortedVel,
             uint  *gridParticleIndex,
             uint  *cellStart,
             uint  *cellEnd,
             uint   numParticles,
             uint   numCells,
             float particleRadius,
             float3 worldOrigin,
             uint3 gridSize,
             float3 cellSize,
             uint *d_neighborList,
             uint MAX_LISTELEM)

{

    
    // thread per particle
    uint numThreads, numBlocks;
    computeGridSize(numParticles, 64, numBlocks, numThreads);
    
    // execute the kernel
    collideD<<< numBlocks, numThreads >>>(//(float4 *)newVel,
                                          (float4 *)sortedPos,
                                          //(float4 *)sortedVel,
                                          gridParticleIndex,
                                          cellStart,
                                          cellEnd,
                                          numParticles,
                                          numCells,
                                          particleRadius,
                                          worldOrigin,
                                          gridSize,
                                          cellSize,
                                          d_neighborList,
                                          MAX_LISTELEM);
    
    // check if kernel invocation generated an error
    getLastCudaError("Kernel execution failed");
    
}
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


void find_neighbor_cuda(int n,double *xp,double *yp,double dis,int **neighborListoutput,int MAX_LIST)
{
    int numParticles=n;

    float particleRadius=dis/2.0f;
    int GRID_SIZE=64;
    
    uint gridDim = GRID_SIZE;
    
    uint3 gridSize;
    gridSize.x = gridSize.y = gridSize.z = gridDim;

    float3 worldOrigin;
    worldOrigin = make_float3(-0.0450f, -0.0450f, -0.0450f);

    float3 cellSize;
    cellSize = make_float3(particleRadius * 2.0f,particleRadius * 2.0f,particleRadius * 2.0f);
    
    uint numCells;
    numCells = gridSize.x*gridSize.y*gridSize.z;
    


 //   uint numBodies;
 //   uint maxParticlesPerCell;
    // CPU data
   
    float *m_hPos=new float[numParticles*4];              // particle positions
   
    

  //  uint  *m_hCellStart;
  //  uint  *m_hCellEnd;
    uint  *m_hGridParticleHash=new uint[numParticles];
    uint  *m_hGridParticleIndex=new uint[numParticles];
    
    // GPU data
    float *m_dPos;
    
    float *m_dSortedPos;
    
    // grid data for sorting method
    uint  *m_dGridParticleHash; // grid hash value for each particle
    uint  *m_dGridParticleIndex;// particle index for each particle
    uint  *m_dCellStart;        // index of start of each cell in sorted list
    uint  *m_dCellEnd;          // index of end of cell


   // float *m_cudaPosVBO;        // these are the CUDA deviceMem Pos

    unsigned int memSize = sizeof(float) * 4 * numParticles;
  
    allocateArray((void **)&m_dPos, memSize);
    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dGridParticleHash, numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, numParticles*sizeof(uint));
    allocateArray((void **)&m_dCellStart, numCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, numCells*sizeof(uint));
    
   // checkCudaErrors(cudaMalloc((void **)&m_cudaPosVBO, memSize)) ;
    
    
    
    //initial particle position
    
    //generate_random_points(m_hPos,numParticles);
    for(int i = 0; i < numParticles; ++i)
    {
        //srand (time(NULL));
        m_hPos[i*4] = xp[i];
        m_hPos[i*4+1] = yp[i];
        m_hPos[i*4+2] = 0;
        m_hPos[i*4+3] = 0;
    }
    
    checkCudaErrors(cudaMemcpy(m_dPos, m_hPos, sizeof(float)*4*numParticles,cudaMemcpyHostToDevice));
    
    // ********************** find particle cell hash id**********************//
   
    calcHash(
             m_dGridParticleHash,
             m_dGridParticleIndex,
             m_dPos,
             numParticles,
             worldOrigin,
             gridSize,
             cellSize);
    
    // ********************** write particle position data**********************//
    
    //checkCudaErrors(cudaMemcpy(m_hGridParticleIndex, m_dGridParticleIndex, sizeof(float)*4*numParticles,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hGridParticleIndex, m_dGridParticleIndex, sizeof(uint)*numParticles,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hGridParticleHash, m_dGridParticleHash, sizeof(uint)*numParticles,cudaMemcpyDeviceToHost));
    if (0) {
        std::cout <<"Index  "<<"position    x   y   z        "<<"Hash"<<std::endl;
        for(int i=0; i<numParticles; i++)
        {
            printf("%.4d,   %.4f,   %.4f,   %.4f,   %.4d\n",m_hGridParticleIndex[i],m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2],m_hGridParticleHash[i]);
        }
    }

    
    
    // ********************** sort particles based on hash **********************//
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, numParticles);
    
    

    
    // ********************** reorder particle arrays into sorted order and**********************//
    // find start and end of each cell
    reorderDataAndFindCellStart(
                                m_dCellStart,
                                m_dCellEnd,
                                m_dSortedPos,
                                //m_dSortedVel,
                                m_dGridParticleHash,
                                m_dGridParticleIndex,
                                m_dPos,
                                //m_dVel,
                                numParticles,
                                numCells);
    
    // ********************** write particle position data reordering index**********************//
    /*
    
  
    checkCudaErrors(cudaMemcpy(m_hGridParticleIndex, m_dGridParticleIndex, sizeof(float)*numParticles,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hGridParticleHash, m_dGridParticleHash, sizeof(float)*numParticles,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hPos, m_dSortedPos, sizeof(float)*4*numParticles,cudaMemcpyDeviceToHost));
    
    std::cout <<"After reordering !!!"<<std::endl;
    std::cout <<"Index  "<<"position    x   y   z        "<<"Hash"<<std::endl;
    for(int i=0; i<numParticles; i++)
    {
        printf("%.4d,   %.4f,   %.4f,   %.4f,   %.4d\n",m_hGridParticleIndex[i],m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2],m_hGridParticleHash[i]);
    }
    
    
    */
    // ********************** write particle position data**********************//

    uint MAX_LISTELEM=MAX_LIST;
    uint *h_neighborList=new uint[numParticles*MAX_LISTELEM];
    uint *d_neighborList;          // index of end of cell
    allocateArray((void **)&d_neighborList, numParticles*MAX_LISTELEM*sizeof(uint));
    cudaMemset(d_neighborList, 0000, MAX_LISTELEM*numParticles*sizeof(uint));
    
    collide(
            //m_dVel,
            m_dSortedPos,
            //m_dSortedVel,
            m_dGridParticleIndex,
            m_dCellStart,
            m_dCellEnd,
            numParticles,
            numCells,
            particleRadius,
            worldOrigin,
            gridSize,
            cellSize,
            d_neighborList,
            MAX_LISTELEM);
    

    checkCudaErrors(cudaMemcpy(m_hGridParticleIndex, m_dGridParticleIndex, sizeof(uint)*numParticles,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hGridParticleHash, m_dGridParticleHash, sizeof(uint)*numParticles,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hPos, m_dSortedPos, sizeof(float)*4*numParticles,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_neighborList, d_neighborList, sizeof(uint)*numParticles*MAX_LISTELEM,cudaMemcpyDeviceToHost));
    if (0) {
        std::cout <<"neibouring index !!!"<<std::endl;
        std::cout <<"i     originalIndex  "<<"position    x   y   z        "<<"Hash"<<std::endl;
        for(int i=0; i<numParticles; i++)
        {
            printf("%.4d,  %.4d,   %.4f,   %.4f,   %.4f,   %.4d,           %.4d,    %.4d    %.4d\n",i,m_hGridParticleIndex[i],m_hPos[i*4],m_hPos[i*4+1],m_hPos[i*4+2],m_hGridParticleHash[i], h_neighborList[i*MAX_LISTELEM],h_neighborList[i*MAX_LISTELEM+1],h_neighborList[(i+1)*MAX_LISTELEM-1]);
        }
    }

    
    
//    uint neighborList[numParticles][MAX_LISTELEM];
    uint** neighborList;
    neighborList=(uint**) malloc(numParticles*sizeof(uint*));
    for (int i = 0; i < numParticles; i++)
        neighborList[i] = (uint*) malloc(MAX_LISTELEM*sizeof(uint));
    for(int i=0;i<numParticles;i++)
    {
        for(int j=0;j<MAX_LISTELEM;j++)
        {
                neighborList[i][j]=0.;
        }
    }
    
    //uint* tempList=new uint[numParticles*MAX_LISTELEM];
    uint* tempList;
    tempList=(uint*) malloc(numParticles*MAX_LISTELEM*sizeof(uint));
    for(int j=0;j<numParticles*MAX_LISTELEM;j++)
    {
        tempList[j]=0.;
    }
    
    //memcpy(neighborList,h_neighborList,numParticles*MAX_LISTELEM*sizeof(uint));
    for (int i=0; i<numParticles; i++) {
        for (int j=0; j<MAX_LISTELEM; j++) {
            neighborList[i][j]=h_neighborList[i*MAX_LISTELEM+j];
        }

    }
    
    
    for (int i=0; i<numParticles; i++) {
        for (int j=0; j<MAX_LISTELEM; j++) {
            tempList[i*MAX_LISTELEM+j]=neighborList[i][j];
        }
    }
    
    
    for (int i=0; i<numParticles; i++) {
        for (int j=0; j<MAX_LISTELEM; j++) {
            neighborListoutput[m_hGridParticleIndex[i]][j]=tempList[i*MAX_LISTELEM+j];
        }
    }
    
    
    

    sortParticles(m_dGridParticleIndex, m_dGridParticleHash, numParticles);
    //checkCudaErrors(cudaMemcpy(m_hGridParticleIndex, m_dGridParticleIndex, sizeof(uint)*numParticles,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(m_hGridParticleHash, m_dGridParticleHash, sizeof(uint)*numParticles,cudaMemcpyDeviceToHost));
    //checkCudaErrors(cudaMemcpy(m_hPos, m_dSortedPos, sizeof(float)*4*numParticles,cudaMemcpyDeviceToHost));
    //checkCudaErrors(cudaMemcpy(h_neighborList, d_neighborList, sizeof(uint)*numParticles*MAX_LISTELEM,cudaMemcpyDeviceToHost));
    
    if (0) {
        std::cout <<"neibouring index !!!"<<std::endl;
        std::cout <<"i      hash id  "<<"neighbor particle id       "<<"total neighbors"<<std::endl;
        for(int i=0; i<numParticles; i++)
        {
            printf("%.4d,   %.4d,      %.4d,   %.4d,   %.4d,   %.4d,      %.4d\n",i,m_hGridParticleHash[i], neighborListoutput[i][0],neighborListoutput[i][1],neighborListoutput[i][2],neighborListoutput[i][3],neighborListoutput[i][MAX_LISTELEM-1]);
        }
    }

    
    

    
    
    free(m_hPos);
    free(m_hGridParticleHash);
    free(m_hGridParticleIndex);
    cudaFree(m_dPos);
    cudaFree(m_dGridParticleHash);
    cudaFree(m_dGridParticleIndex);
        cudaFree(m_dCellStart);
        cudaFree(m_dCellEnd);
        cudaFree(m_dSortedPos);
    
    free(h_neighborList);
    cudaFree(d_neighborList);
    free(tempList);
    free(neighborList);
    
    //std::cout<<"sorting finished !!!"<<std::endl;
    return;
    
}
}   // extern "C"
#endif // PARTICLE_H