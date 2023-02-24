#include <iostream>
#include "src/simreader.h"
#include "src/simulation.h"
#include "src/particle.h"
#include "vector"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "cuda_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include <cstring>
#include "curand.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"

using std::cout;
using std::endl;


__global__ void setup_kernel (curandStatePhilox4_32_10_t *state, unsigned long seed)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void sum_array_gpu(double * a, double * b, double * c, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}

}



__global__ void doaflip(double * A, double * Bounds, int * LUT,bool * conditionalExample, curandStatePhilox4_32_10_t *state, int size)
{
	int gid = threadIdx.x + blockDim.x * blockIdx.x;



//	int stride = blockDim.x*gridDim.x;

	if (gid < size)
	{
		if (gid < 3)
		{
			printf("Bounds: [%.4f %.4f %.4f]\n",Bounds[0],Bounds[1],Bounds[2]);
		}
		curandStatePhilox4_32_10_t localstate = state[gid];
		double4 xi;
		xi = curand_uniform4_double(&localstate);
		printf("Xi: [%f %f %f]\n",xi.x,xi.y,xi.z);
		printf("Xi*b: [%f %f %f]\n",xi.x*Bounds[0],xi.y*Bounds[1],xi.z*Bounds[2]);
		A[3*gid+0]=xi.x*Bounds[0];
		A[3*gid+1]=xi.y*Bounds[1];
		A[3*gid+2]=xi.z*Bounds[2];

		int _A[3];

		_A[0] = (int) A[3*gid+0];
		_A[1] = (int) A[3*gid+1];
		_A[2] = (int) A[3*gid+2];

		printf("Floor of A: [%d %d %d]\n", _A[0],_A[1],_A[2]);
		int b[3];
		b[0] = (int) Bounds[0];
		b[1] = (int) Bounds[1];
		b[2] = (int) Bounds[2];

		//		int linearIndex = offset + bz*(by*i + j) + k;
		int linearIndex = 0 + b[2]*(b[1]*_A[0] + _A[1]) + _A[2];

		printf("Linear Index: %d\t Value of Lookup: %d \n", linearIndex, LUT[linearIndex]);


		conditionalExample[gid] = (bool) LUT[linearIndex];

		printf("conditionalExample[gid]: %d\n", conditionalExample[gid]);





	}

}



int main()
{
    std::string path = "/autofs/homes/009/bs244/cuda-workspace/hellocuda/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);
    sim.setStep_num(1000);
    sim.setParticle_num(10000);
    double * stepsize =  sim.getStep_size();
    particle par(stepsize);
    par.display();
    par.setFlag();
    par.setState();
    double position[3] = {1,2,3};
    par.setPos(position);
    par.setState(false);
    par.display();

    sim.setParticle_num(100);
    double poses[int(3*sim.getParticle_num())];
    double * nextPositions = sim.nextPosition(poses);

    for (int i = 0; i < 3 * int(sim.getParticle_num()); i+=3)
    {
//        printf("particle: %.3f %.3f %.3f\n",poses[i],poses[i+1],poses[i+2]);
    }

    double nextvectors[int(3*sim.getParticle_num())];
    double *coords;
    auto elapsed = clock();
    time_t time_req;
    time_req = clock();
    for (int i = 0; i < 10; i++)
    {
        time_req = clock();
        coords = sim.nextPosition(nextvectors);
        for (int j = 0; j<int(3*sim.getParticle_num()); j+=3)
        {
            printf("Sum:\n%.3f + %.3f | %.3f\n%.3f + %.3f | %.3f\n%.3f + %.3f | %.3f\n\n",nextPositions[j+0],coords[j+0],nextPositions[j+0]+coords[j+0],nextPositions[j+1],coords[j+1],nextPositions[j+1]+coords[j+1],nextPositions[j+2],coords[j+2],nextPositions[j+2]+coords[j+2]);
            nextPositions[j+0] = nextPositions[j+0] + coords[j+0];
            nextPositions[j+1] = nextPositions[j+1] + coords[j+1];
            nextPositions[j+2] = nextPositions[j+2] + coords[j+2];
            printf("Position After:\t[%.3f, %.3f, %.3f]\n",nextPositions[j+0],nextPositions[j+1],nextPositions[j+2]);
        }
        time_req = clock()-time_req;
        std::cout << std::endl << (float)time_req/CLOCKS_PER_SEC << " seconds" << std::endl;
    }



    int size = 100;
    int block_size = 128;
    int NO_BYTES = size * sizeof(double);

//    host pointers

    double * hostA;
    double * hostB;
    double * hostResults;

//   alloc memory for host pointers
    hostA = (double*)malloc(NO_BYTES);
    hostB = (double*)malloc(NO_BYTES);

    hostResults = (double*) malloc(NO_BYTES);

//    init host pointer
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i< int(size); i++)
    {
       	hostA[i] = ((rand() &0xFF)/(double) i);
    }
    for (int i = 0; i< int(size); i++)
    {
       	hostB[i] = ((rand() &0xFF)/(double) i);
       	srand((unsigned) time(&t));
    }

    memset(hostResults,0,NO_BYTES);

    // device pointer

    double * deviceA;
    double * deviceB;
    double * deviceResults;

    cudaMalloc((double ** )&deviceA, NO_BYTES);
    cudaMalloc((double ** )&deviceB, NO_BYTES);
    cudaMalloc((double ** )&deviceResults, NO_BYTES);


    // memory transfer host -> device
    cudaMemcpy(deviceA,hostA,NO_BYTES,cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB,hostB,NO_BYTES,cudaMemcpyHostToDevice);

    //launch grid

    dim3 block(block_size);
    dim3 grid((size/block.x)+1);

    sum_array_gpu<<<grid,block>>>(deviceA,deviceB,deviceResults,size);
    cudaDeviceSynchronize();
    //transfer results back to host
    cudaMemcpy(hostResults,deviceResults,NO_BYTES,cudaMemcpyDeviceToHost);
    cudaFree(deviceResults);
    cudaFree(deviceB);
    cudaFree(deviceA);

//    // check results
//    for (int m = 0; m<size; m++)
//    {
//    	printf("results host array[%d]: %.4f + %.4f = %.4f\n",m,hostA[m],hostB[m],hostResults[m]);
//    }

    free(hostResults);
    free(hostA);
    free(hostB);

    int random_bytes = size * sizeof(double) * 3;

    double * h_a;
    double * hostBounds;
    int * hostLookup;
    bool * hostLogicalVector;

//   alloc memory for host pointers
    h_a = (double*)malloc(random_bytes);
    hostBounds = (double*)malloc(3*sizeof(double));
    hostLogicalVector = (bool*)malloc(size*sizeof(bool));

    hostBounds[0] = 5.0; hostBounds[1] = 5.0; hostBounds[2] = 5.0;

    /**
     * x,y,z -> r,c,p
     * 			Page
     * 			Col
     *     	Row	[0 0 0] | 		[0 0 0] | [0 0 0]
     *     	 	[0 1 0] |		[0 1 0] | [0 1 0]
     *     		[0 0 0] | 		[0 0 0] | [0 0 0]
     *
     *
     *     		(1,1,0) (1,1,1) (1,1,2)
     */
    int prod = (int) (hostBounds[0] * hostBounds[1] * hostBounds[2]);
    hostLookup = (int*)malloc(prod*sizeof(int));

    double * d_a;
    double * deviceBounds;
    int * deviceLookup;

    bool * deviceLogicalVector;




    cudaMalloc((int **)&deviceLookup, prod*sizeof(int));
    cudaMalloc((double ** )&d_a, random_bytes);
    cudaMalloc((double ** )&deviceBounds,3*sizeof(double));
    cudaMalloc((bool ** )&deviceLogicalVector,size*sizeof(bool));


	cudaMemcpy(deviceBounds,hostBounds,3*sizeof(double), cudaMemcpyHostToDevice);
    curandStatePhilox4_32_10_t * deviceState;
    memset(hostLookup,0,prod*sizeof(int));
    memset(hostLogicalVector,false,size*sizeof(bool));


    int bx = (int) (hostBounds[0]);
    int by = (int) (hostBounds[1]);
    int bz = (int) (hostBounds[2]);

    printf ("bx, by, bz: %d, %d, %d\n", bx, by,bz);
    int id0 = 0 + bx*(by*0 + 1) + 1;
    int id1 = 0 + bx*(by*1 + 1) + 1;
    int id2 = 0 + bx*(by*2 + 1) + 1;
    int id3 = 0 + bx*(by*3 + 1) + 1;
    int id4 = 0 + bx*(by*4 + 1) + 1;
    int id01 = 0 + bx*(by*0 + 2) + 1;
    int id11 = 0 + bx*(by*1 + 2) + 1;
    int id21 = 0 + bx*(by*2 + 2) + 1;
    int id31 = 0 + bx*(by*3 + 2) + 1;
    int id41 = 0 + bx*(by*4 + 2) + 1;

    hostLookup[id0] = 1;
    hostLookup[id1] = 1;
    hostLookup[id2] = 1;
    hostLookup[id3] = 1;
    hostLookup[id4] = 1;
    hostLookup[id01] = 1;
    hostLookup[id11] = 1;
    hostLookup[id21] = 1;
    hostLookup[id31] = 1;
    hostLookup[id41] = 1;
    cudaMemcpy(deviceLookup,hostLookup,prod*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLogicalVector,hostLogicalVector,size*sizeof(bool),cudaMemcpyHostToDevice);

    for(int i=0; i<bx; i++)
    {

    	for (int j=0; j<by; j++)
		{
    		for (int k=0; k<bz; k++)
			{
    			int idx = 0 + bz*(by*i + j) + k;
    			printf("[i,j,k]: [%d %d %d]\t",i,j,k);
    			printf("linear index: %d\t\t", idx);
    			printf("hostLookUp[i,j,k]: %d\n", hostLookup[idx]);
			}
		}
    }



    cudaMalloc((curandStatePhilox4_32_10_t**)&deviceState,size*sizeof(curandStatePhilox4_32_10_t));

    setup_kernel<<<grid, block>>>(deviceState,1);

    doaflip<<<grid, block>>>(d_a,deviceBounds, deviceLookup, deviceLogicalVector,deviceState,size);
	cudaDeviceSynchronize();

	cudaMemcpy(h_a,d_a,random_bytes,cudaMemcpyDeviceToHost);
	cudaMemcpy(hostLogicalVector,deviceLogicalVector,size*sizeof(bool),cudaMemcpyDeviceToHost);
//	cudaMemcpy(hostBounds, deviceBounds, 3*sizeof(double), cudaMemcpyDeviceToHost);


	for (int i = 0; i<size; i++)
	{
		double x = h_a[3*i+0];
		double y = h_a[3*i+1];
		double z = h_a[3*i+2];
		printf("x: %.4f \t y: %.4f \t z: %.4f\t Has Lookup Values: %d\n",x,y,z,hostLogicalVector[i]);
	}

	cudaFree(d_a);
	cudaFree(deviceBounds);
	cudaFree(deviceState);

	free(h_a);
	free(hostBounds);

    return 0;

}
