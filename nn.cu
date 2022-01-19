#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

__global__ void mykernel(int *a, int *b, int *c) {
  __shared__ int temp[...];
  int i = threadIdx.x;

  __syncthreads(); // sync threads between staging data and doing computation?

  c[i] = a[i] + b[i];
}


int main(void) {
  int i, j;

  /* MPI */
  int nproc=1, myproc=0;

  /* Compute ground truth */
  const float y[] = {0,0,1.,1.,-1.,1.,-1.,1.,-1.,-1.,-1.,1.,-1.,1.,-1.,-1.};
  unsigned char num;
  float in[16][5];
  for (num=2; num<16; ++num) {
    in[num][0] = 1.;
    for (i=1; i<5; ++i) in[num][i] = num & 1<<i ? 1. : 0.;
  }
  if (1) for (num=2; num<16; ++num) {
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

    // in principle should randomize order but seems like too much trouble
    

#if 0

    Application can query and select GPUs:
cudaGetDeviceCount(int *count)
cudaSetDevice(int device)
cudaGetDevice(int *device)
cudaGetDeviceProperties(cudaDeviceProp *prop, int device)

  cudaMemcpy() can copy from one device to another

  cudaMalloc(&da, size);
  cudaMalloc(&db, size);
  cudaMalloc(&dsum, size);



  for (int i=0; i<N; ++i) {
    a[i] = i;
    b[i] = 2*i+1;
  }

  cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

  mykernel<<<1,N>>>(da, db, dsum);

  // Does each block somehow finish separately, and return here?  Doesn't make sense to me.  This code is serial as far as CUDA is concerned, no??
  // I think he is imagining that the host code is multithreaded.

  // Not sure that is it either?  Below barrier waits for all "preceding CUDA calls to complete"
cudaDeviceSynchronize()
  // cudaMemcpy() begins and ends with implicit cudaDeviceSynchronize()
  // (else use cudaMemcpyAsync())

  cudaError_t cerr = cudaGetLastError();
  char *cerrst = cudaGetErrorString(cerr);
  or better:
  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  //printf("Hello World! %d %d %d %d\n", a, b, c, d);

  cudaMemcpy(sum, dsum, size, cudaMemcpyDeviceToHost);
  //cudaMemcpy(&d, dd, sizeof(int), cudaMemcpyDeviceToHost);

  //printf("Hello World! %d %d %d %d\n", a, b, c, d);
  for (int i=0; i<N; ++i) printf("%d + %d = %d\n", a[i], b[i], sum[i]);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dsum);
#endif

  }

  return 0;
}
