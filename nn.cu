#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

#define VERB 1
#define DEBUG 1
#define NTHREADS NBITS+1 // I don't need below
// (NBITS+1>NNEURONS ? NBITS+1 : NNEURONS) // max num inputs per neuron

/* Check for CUDA error */
 void cudaerr(const char *msg) {
   cudaError_t cerr = cudaGetLastError();
   //char *cerrst = cudaGetErrorString(cerr);
   //or better:
   if (cerr) printf("CUDA error from %s : %s\n", msg, cudaGetErrorString(cerr));
 }

#ifdef DEBUG
#define DI(i,j) ((NBITS+1)*i+j)
__global__ void learn(float *truth, float *in, float *a1, float *a2, float *out, float *debug) {
#else
__global__ void learn(float *truth, float *in, float *a1, float *a2, float *out) {
#endif
  __shared__ float temp[NNEURONS+1][NTHREADS];
  __shared__ float z[NNEURONS+1]; // FIX: +1 offset will be confusing!
  int ti = threadIdx.x; // index of dot product
  int ni = threadIdx.y; // index of neuron

  /* Forward Propagate */
  /* Layer 1 */

  /* Compute inner product */
  if (ni!=NNEURONS) temp[ni][ti] = a1[AI(ni,ti)] * in[ti];
#ifdef DEBUG
  debug[DI(ni,ti)] = temp[ni][ti];
#endif
  __syncthreads();
  if (!ti) {
    z[0] = 1.;
    // z[1..NNEURONS] z inner product on each neuron
    z[ni+1] = temp[ni][0];
    for (int i=1; i<NBITS+1; ++i) z[ni+1] += temp[ni][i];
  }
#ifdef DEBUG
  else z[1] = ti;
  //debug[DI(ni,ti)] = z[1]; correct inner product of 1st neuron
#endif
  __syncthreads();

  /* Layer 2 */
  // z[0] = 1 z[i] = z[neuron+1]

  /* Compute inner product */
  if (!ti) temp[ni][0] = a2[ni] * z[ni];
  __syncthreads();
  if (!ti && !ni) {
    *out = temp[0][0];
    for (int i=1; i<NNEURONS+1; ++i) *out += temp[i][0];
  }
#ifdef DEBUG
  //debug[DI(ni,ti)] = z[2]; // a2[ni]; //temp[1][0];
#endif
}


int main(void) {
  int i, j;
  const int nnum = 1<<NBITS;
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
    for (i=1; i<NBITS+1; ++i) in[num][i] = num & 1<<(i-1) ? 1. : 0.;
  }
  if (VERB) for (num=2; num<nnum; ++num) {
    printf("%2d %3.0f : ", num, y[num]);
    printf("%2.f %2.f %2.f %2.f\n", in[num][0], in[num][1], in[num][2],
	   in[num][3]);
  }

  /* Define neural network */
  if (DEBUG) srand48(0);
  float *a1, *a2;
  a1 = (float *) malloc((NBITS+1)*NNEURONS*sf);
  //z1 = (float *) malloc(NNEURONS*sf);
  a2 = (float *) malloc((NNEURONS+1)*sf);
  for (i=0; i<NNEURONS; ++i) for (j=1; j<NBITS+1; ++j) {
    a1[AI(i,0)] = 0.; // start biases at 0
    a1[AI(i,j)] = 2*drand48()-1;
  }
  //for (i=0; i<NNEURONS; ++i) z1[i] = 2*drand48()-1;
  a2[0] = 0.;
  for (i=1; i<NNEURONS+1; ++i) a2[i] = 2*drand48()-1;
  float *out; // output value
  out = (float *) malloc(sf); // stack seems okay too

#ifdef DEBUG
  for (i=0; i<NNEURONS; ++i) {
    printf("\na1[%d] =", i);
    for (j=0; j<NBITS+1; ++j) printf("%10f", a1[AI(i,j)]);
  }
  printf("\na2    =");
  for (i=0; i<NNEURONS+1; ++i) printf("%10f", a2[i]);
  printf("\n");
  //float debug[NNEURONS][NBITS+1], *ddebug;
  float *debug, *ddebug;
  debug = (float *) malloc(sf*(NBITS+1)*(NNEURONS+1));
  //memset(debug, 'x', sf*(NBITS+1)*(NNEURONS+1));
  for (i=0; i<NNEURONS+1; ++i) {
    printf("nurn %d:", i);
    for (j=0; j<NBITS+1; ++j) {
      debug[DI(i,j)] = -9.;
      printf(" %9f", debug[DI(i,j)]);
    }
    printf("\n");
  }
  printf("-----------------------------------\n");
  cudaMalloc(&ddebug, sf*(NBITS+1)*(NNEURONS+1));
#endif

  float *dtruth, *din, *da1, *da2, *dout;
  cudaMalloc(&dtruth, sf);
  cudaMalloc(&din, sf*(NBITS+1));
  cudaMalloc(&da1, sf*(NBITS+1)*NNEURONS);
  cudaMalloc(&da2, sf*(NNEURONS+1));
  cudaMalloc(&dout, sf);
  
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
      printf("num = %d :\n", num);
      //      for (i=0; i<

      /* Stage nn computation on GPU */
      cudaMemcpy(dtruth, &y[num], sf, cudaMemcpyHostToDevice);
      cudaMemcpy(din, in[num], (NBITS+1)*sf, cudaMemcpyHostToDevice);
      cudaMemcpy(da1, a1, (NBITS+1)*NNEURONS*sf, cudaMemcpyHostToDevice);
      cudaMemcpy(da2, a2, (NNEURONS+1)*sf, cudaMemcpyHostToDevice);

      /* Run the kernel */
      //dim3 threads(NTHREADS,NNEURONS);
      dim3 threads(NBITS+1,NNEURONS+1);
#ifdef DEBUG
      learn<<<1,threads>>>(dtruth, din, da1, da2, dout, ddebug);
#else
      learn<<<1,threads>>>(dtruth, din, da1, da2, dout);
#endif
      cudaerr("kernel");

      /* Copy results back to host */
      cudaMemcpy(out, dout, sf, cudaMemcpyDeviceToHost);
      cudaerr("out copy to host");
#ifdef DEBUG
      cudaMemcpy(debug, ddebug, sf*(NBITS+1)*(NNEURONS+1), cudaMemcpyDeviceToHost);
      cudaerr("debug copy to host");
      for (i=0; i<NNEURONS+1; ++i) {
	printf("nurn %d:", i);
	for (j=0; j<NBITS+1; ++j) printf(" %9f", debug[DI(i,j)]);
	printf("\n");
      }
#endif

      printf("NN output: %f\n", *out);
      //for (int i=0; i<N; ++i) printf("%d + %d = %d\n", a[i], b[i], sum[i]);
      return 2;

    } // loop over data
  } // training loop
  
  cudaFree(dtruth);
  cudaFree(din);
  cudaFree(da1);
  cudaFree(da1);
  cudaFree(da1);

  return 0;
}
