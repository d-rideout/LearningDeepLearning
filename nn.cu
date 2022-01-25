#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "nn.h"

#define VERB 1
//#define DEBUG 1
#define NDEBUG 5
#define DI(h,i,j) (h*(NNEURONS+1)*(NBITS+1)+(NBITS+1)*i + j)
#define NTHREADS NBITS+1 // I don't need below
// (NBITS+1>NNEURONS ? NBITS+1 : NNEURONS) // max num inputs per neuron

/* Check for CUDA error */
void cudaerr(const char *msg) {
  cudaError_t cerr = cudaGetLastError();
  //char *cerrst = cudaGetErrorString(cerr);
  //or better:
  if (cerr) printf("CUDA error from [%s] : %s\n", msg, cudaGetErrorString(cerr));
}

/* Output weights */
void display_weights(float *a1, float *a2) {
  unsigned char i=1;
  if (DEBUG) i=0;
  for (; i<NNEURONS+1; ++i) {
    printf("a1[%d] =", i);
    for (unsigned char j=0; j<NBITS+1; ++j) printf("%10f", a1[AI(i,j)]);
    printf("\n");
  }
  printf("a2    =");
  for (i=0; i<NNEURONS+1; ++i) printf("%10f", a2[i]);
  printf("\n");
}


#if DEBUG==1
__global__ void learn(float *truth, float *in, float *a1, float *a2, float *out, float *debug) {
#else
__global__ void learn(float *truth, float *in, float *a1, float *a2, float *out) {
#endif
  __shared__ float temp[NNEURONS+1][NBITS+1];
  __shared__ float z[NNEURONS+1];
  int ti = threadIdx.x; // index of dot product
  int ni = threadIdx.y; // index of neuron

  /* Forward Propagate */
  /* Layer 1 */

  /* Compute inner product */
  //if (ni!=NNEURONS) 
  //if (ni) 
  temp[ni][ti] = a1[AI(ni,ti)] * in[ti];
    //else temp[ni][ti] = -99.;
#if DEBUG==1
  debug[DI(0,ni,ti)] = temp[ni][ti];
#endif
  __syncthreads();

  z[0] = 1.;
  // z[1..NNEURONS] z inner product on each neuron
  if (ni>0) {
    z[ni] = temp[ni][0];
    for (int i=1; i<NBITS+1; ++i) z[ni] += temp[ni][i];
  }

#if DEBUG==1
  //else z[1] = ti;
  debug[DI(1,ni,ti)] = z[1]; //correct inner product of 1st neuron
#endif
  __syncthreads();

  /* Layer 2 */
  // z[0] = 1 z[i] = z[neuron+1]

  /* Compute inner product */
  //if (!ti) 
  temp[ni][0] = a2[ni] * z[ni];
  __syncthreads();
#if DEBUG==1
  debug[DI(2,ni,ti)] = temp[ni][0];
#endif
  //if (!ti && !ni) {
    *out = temp[0][0];
    for (int i=1; i<NNEURONS+1; ++i) *out += temp[i][0];
    __syncthreads();
    //}
#if DEBUG==1
    debug[DI(3,ni,ti)] = temp[1][0];
#endif

  //assert(*out);
  if (*truth * *out > 0) return; // signs agree

  /* Back Propagate */
  // derivative of error = truth-output
  float de = *out - *truth; // > 0 ? -1. : 1; //sign(-*out);
  // error of layer 2 neuron
  float err2 = de;

  // Error of each neuron in layer 1
  temp[ni][ti] = err2*a2[ni]; // error of neuron ni
  
  /* Adjust weights */
#if DEBUG==1
  debug[DI(4,ni,ti)] = temp[ni][ti]; // ETA*in[ti]*temp[ni][ti];
#endif
  if (!ti) a2[ni] -= ETA*z[ni]*err2;
  a1[AI(ni,ti)] -= ETA*in[ti]*temp[ni][ti];
  //__syncthreads();
  //debug[DI(ni,ti)] = a2[ni]; //err2; //z[ni]; //temp[ni][ti]; // ETA*in[ti]*temp[ni][ti];
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
  a1 = (float *) malloc((NBITS+1)*(NNEURONS+1)*sf);
  //z1 = (float *) malloc(NNEURONS*sf);
  a2 = (float *) malloc((NNEURONS+1)*sf);
  for (j=0; j<NBITS+1; ++j) a1[AI(0,j)] = 0.; // initialize 0-neuron to 0
  for (i=1; i<NNEURONS+1; ++i) for (j=1; j<NBITS+1; ++j) {
    a1[AI(i,0)] = 0.; // start biases at 0
    if (DEBUG) a1[AI(i,j)] = i + .1*j;
    else a1[AI(i,j)] = 2*drand48()-1;
  }
  //for (i=0; i<NNEURONS; ++i) z1[i] = 2*drand48()-1;
  a2[0] = 0.;
  for (i=1; i<NNEURONS+1; ++i) if (DEBUG) a2[i] = -(i*.1+.01);
    else a2[i] = 2*drand48()-1;
  float *out; // output value
  out = (float *) malloc(sf); // stack seems okay too

#if DEBUG==1
  int h;
  display_weights(a1, a2);

  //float debug[NNEURONS][NBITS+1], *ddebug;
  float *debug, *ddebug;
  debug = (float *) malloc(sf*(NBITS+1)*(NNEURONS+1)*NDEBUG);
  //memset(debug, 'x', sf*(NBITS+1)*(NNEURONS+1));
  printf("-----------------------------------\n");
  for (h=0; h<NDEBUG; ++h) for (i=0; i<NNEURONS+1; ++i) {
    printf("nurn %d:", i);
    for (j=0; j<NBITS+1; ++j) {
      debug[DI(h,i,j)] = -h;
      printf(" %9f", debug[DI(h,i,j)]);
    }
    printf("\n");
  }
  printf("-----------------------------------\n");
  cudaMalloc(&ddebug, sf*(NBITS+1)*(NNEURONS+1)*NDEBUG);
#endif

  float *dtruth, *din, *da1, *da2, *dout;
  cudaMalloc(&dtruth, sf);
  cudaMalloc(&din, sf*(NBITS+1));
  cudaMalloc(&da1, sf*(NBITS+1)*(NNEURONS+1));
  cudaMalloc(&da2, sf*(NNEURONS+1));
  cudaMalloc(&dout, sf);
  
  /* Training loop */
  unsigned int sweep=0;
  unsigned char nwrong=1;
  
  while (nwrong) {
    if (++sweep > MAX_SWEEP) {
      printf("Too many sweeps.\n");
      break;
    }
    printf("Sweep %u\n", sweep);
    nwrong = 0;
    //++sweep;

    /* Loop over data */
    // in principle should randomize order but seems like too much trouble
    for (num=2; num<1<<NBITS; ++num) {
      printf("num = %d :\n", num);
      //      for (i=0; i<

      /* Stage nn computation on GPU */
      cudaMemcpy(dtruth, &y[num], sf, cudaMemcpyHostToDevice);
      cudaMemcpy(din, in[num], (NBITS+1)*sf, cudaMemcpyHostToDevice);
      cudaMemcpy(da1, a1, (NBITS+1)*(NNEURONS+1)*sf, cudaMemcpyHostToDevice);
      cudaMemcpy(da2, a2, (NNEURONS+1)*sf, cudaMemcpyHostToDevice);

      /* Run the kernel */
      //dim3 threads(NTHREADS,NNEURONS);
      dim3 threads(NBITS+1,NNEURONS+1);
#if DEBUG==1
      learn<<<1,threads>>>(dtruth, din, da1, da2, dout, ddebug);
#else
      learn<<<1,threads>>>(dtruth, din, da1, da2, dout);
#endif
      cudaerr("kernel");

      /* Copy results back to host */
      cudaMemcpy(out, dout, sf, cudaMemcpyDeviceToHost);
      cudaerr("out copy to host");
      cudaMemcpy(a1, da1, (NBITS+1)*(NNEURONS+1)*sf, cudaMemcpyDeviceToHost);
      cudaerr("a1 copy to host");
      cudaMemcpy(a2, da2, (NNEURONS+1)*sf, cudaMemcpyDeviceToHost);
      cudaerr("a2 copy to host");
#if DEBUG==1
      cudaMemcpy(debug, ddebug, sf*(NBITS+1)*(NNEURONS+1)*NDEBUG, cudaMemcpyDeviceToHost);
      cudaerr("debug copy to host");
      for (h=0; h<NDEBUG; ++h) for (i=0; i<NNEURONS+1; ++i) {
	  printf("%d nurn %d:", h, i);
	for (j=0; j<NBITS+1; ++j) printf(" %9f", debug[DI(h,i,j)]);
	printf("\n");
      }
#endif

      printf("NN output: %f\n", *out);

      if (*out*y[num] < 0) {
	display_weights(a1, a2);
	++nwrong;
      }

      if (DEBUG) return 2;

    } // loop over data
    printf("%u wrong\n\n", nwrong);
  } // training loop

  printf("Finished.\n");
  cudaFree(dtruth);
  cudaFree(din);
  cudaFree(da1);
  cudaFree(da2);
  cudaFree(out);
#if DEBUG==1
  cudaFree(debug);
#endif

  return 0;
}
