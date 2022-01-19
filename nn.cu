#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

#define VERB 1


__global__ void learn(float *truth, float *a1, float *a2) {

  //__shared__ int temp[...];
  //int i = threadIdx.x;

  *truth += .5;

  //__syncthreads(); // sync threads between staging data and doing computation?

  //c[i] = a[i] + b[i];
}


int main(void) {
  int i; //, j;

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
  float in[16][5];
  for (num=2; num<16; ++num) {
    in[num][0] = 1.;
    for (i=1; i<5; ++i) in[num][i] = num & 1<<i ? 1. : 0.;
  }
  if (VERB) for (num=2; num<16; ++num) {
    printf("%2d %3.0f : ", num, y[num]);
    printf("%2.f %2.f %2.f %2.f\n", in[num][0], in[num][1], in[num][2],
	   in[num][3]);
  }

  /* Define neural network */
  float *a1, *z1, *a2, z2;
  a1 = (float *) malloc((NBITS+1)*NNEURONS*sizeof(float));
  z1 = (float *) malloc(NNEURONS*sizeof(float));
  a2 = (float *) malloc(NNEURONS*sizeof(float));
  for (i=0; i<(NBITS+1)*NNEURONS; ++i) a1[i] = 2*drand48()-1;
  for (i=0; i<NNEURONS; ++i) z1[i] = 2*drand48()-1;
  for (i=0; i<NNEURONS; ++i) a2[i] = 2*drand48()-1;
  
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
      float *dtruth, *da1, *da2;
      cudaMalloc(&dtruth, sizeof(float));
      cudaMalloc(&da1, sizeof(float)*(NBITS+1)*NNEURONS);
      cudaMalloc(&da2, sizeof(float)*NNEURONS);
      cudaMemcpy(dtruth, &y[num], sizeof(float), cudaMemcpyHostToDevice);
      //cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

      /* Run the kernel */
      learn<<<1,1>>>(dtruth, da1, da2);

      /* See how it went */
      cudaError_t cerr = cudaGetLastError();
      //char *cerrst = cudaGetErrorString(cerr);
      //or better:
      if (cerr) printf("CUDA error string: [%s]\n", cudaGetErrorString(cerr));

      /* Copy results back to host */
      float junk;
      cudaMemcpy(&junk, dtruth, sizeof(float), cudaMemcpyDeviceToHost);
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
