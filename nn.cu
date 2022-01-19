#include <stdio.h>
#include "nn.h"


__global__ void mykernel(int *a, int *b, int *c) {
  int i = threadIdx.x;

  c[i] = a[i] + b[i];
}


int main(void) {
  const float y[] = {0,0,1.,1.,-1.,1.,-1.,1.,-1.,-1.,-1.,1.,-1.,1.,-1.,-1.};
  int i;
  unsigned char num;
  float in[16][4];

  /* Compute ground truth */
  for (num=2; num<16; ++num)
    for (i=0; i<4; ++i) in[num][i] = num & 1<<i ? 1. : 0.;
  for (num=2; num<16; ++num) {
    printf("%2d %3.0f : ", num, y[num]);
    printf("%2.f %2.f %2.f %2.f\n", in[num][0], in[num][1], in[num][2], in[num][3]);
  }

#if 0

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

  //printf("Hello World! %d %d %d %d\n", a, b, c, d);

  cudaMemcpy(sum, dsum, size, cudaMemcpyDeviceToHost);
  //cudaMemcpy(&d, dd, sizeof(int), cudaMemcpyDeviceToHost);

  //printf("Hello World! %d %d %d %d\n", a, b, c, d);
  for (int i=0; i<N; ++i) printf("%d + %d = %d\n", a[i], b[i], sum[i]);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dsum);
#endif

  return 0;
}