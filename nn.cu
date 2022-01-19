#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

#define VERB 1
#define NTHREADS (NBITS+1>NNEURONS ? NBITS+1 : NNEURONS) // max num inputs per neuron

__global__ void learn(float *truth, float *in, float *a1, float *a2) {
  __shared__ int temp[NNEURONS][NTHREADS];
  int i = threadIdx.x; // index of dot product
  int j = threadIdx.y; // index of neuron

  /* Forward Propagate */
  /* Layer 1 */

  /* Populate local memory */
  // if (i<NBITS+1) temp[j,i] = a1[AI(j,i)];

  /* Compute inner product */
  if (i<NBITS+1) temp[j][i] = a1[AI(j,i)] * in[i];
  __syncthreads(); // sync threads between staging data and doing computation?
  
  


}


int main(void) {
  int i; //, j;
  const int nnum = (1<<NBITS)+1;
  const int sf = sizeof(float);
  //  const int nthreads = NBITS+1>NNEURONS ? NBITS+1 : NNEURONS;

  /* MPI */
  //int nproc=1, myproc=0;

  /* GPU */
  int ngpu, mygpu;
  // cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
  cudaGetDeviceCount(&ngpu);
  cudaGetDevice(&mygpu);
  printf("GPU %d of %d GPUs\n", mygpu, ngpu);
  // cudaSetDevice(int device)

  /* Compute ground truth */
  const float y[] = {0,0,1.,1.,-1.,1.,-1.,1.,-1.,-1.,-1.,1.,-1.,1.,-1.,-1.};
  unsigned char num;
  float in[nnum][NBITS+1];
  for (num=2; num<nnum; ++num) {
    in[num][0] = 1.;
    for (i=1; i<NBITS+1; ++i) in[num][i] = num & 1<<i ? 1. : 0.;
  }
  if (VERB) for (num=2; num<nnum; ++num) {
    printf("%2d %3.0f : ", num, y[num]);
    printf("%2.f %2.f %2.f %2.f\n", in[num][0], in[num][1], in[num][2],
	   in[num][3]);
  }

  /* Define neural network */
  //float *a1, *z1, *a2, z2;
  float *a1, *a2;
  a1 = (float *) malloc((NBITS+1)*NNEURONS*sf);
  //z1 = (float *) malloc(NNEURONS*sf);
  a2 = (float *) malloc(NNEURONS*sf);
  a1[0] = 0.; // start biases at 0
  for (i=1; i<(NBITS+1)*NNEURONS; ++i) a1[i] = 2*drand48()-1;
  //for (i=0; i<NNEURONS; ++i) z1[i] = 2*drand48()-1;
  a2[0] = 0.;
  for (i=1; i<NNEURONS; ++i) a2[i] = 2*drand48()-1;
  
  /* Training loop */
  unsigned int sweep=0;
  unsigned char nwrong=1;
  
  while (nwrong) {
    if (sweep > MAX_SWEEP) return 1;
    nwrong = 0;
    ++sweep;

    /* Loop over data */
    // in principle should randomize order but seems like too much trouble
    for (num=2; num<1<<NBITS; ++num) {

      /* Stage nn computation on GPU */
      float *dtruth, *din, *da1, *da2;
      cudaMalloc(&dtruth, sf);
      cudaMalloc(&din, sf*(NBITS+1));
      cudaMalloc(&da1, sf*(NBITS+1)*NNEURONS);
      cudaMalloc(&da2, sf*NNEURONS);

      cudaMemcpy(dtruth, &y[num], sf, cudaMemcpyHostToDevice);
      cudaMemcpy(din, in[num], (NBITS+1)*sf, cudaMemcpyHostToDevice);
      cudaMemcpy(da1, a1, (NBITS+1)*NNEURONS*sf, cudaMemcpyHostToDevice);
      cudaMemcpy(da2, a2, NNEURONS*sf, cudaMemcpyHostToDevice);

      /* Run the kernel */
      dim3 threads(NTHREADS,NNEURONS);
      learn<<<1,threads>>>(dtruth, din, da1, da2);



      /* See how it went */
      cudaError_t cerr = cudaGetLastError();
      //char *cerrst = cudaGetErrorString(cerr);
      //or better:
      if (cerr) printf("CUDA error string: [%s]\n", cudaGetErrorString(cerr));

      /* Copy results back to host */
      float junk;
      cudaMemcpy(&junk, dtruth, sf, cudaMemcpyDeviceToHost);
  //cudaMemcpy(&d, dd, sizeof(int), cudaMemcpyDeviceToHost);

      printf("Hello World! %f\n", junk);
      //for (int i=0; i<N; ++i) printf("%d + %d = %d\n", a[i], b[i], sum[i]);

      //  cudaFree(da);
      //  cudaFree(db);
      //  cudaFree(dsum);

    } // loop over data
  } // training loop

  return 0;
}
