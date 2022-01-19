#include <stdio.h>

__global__ void mykernel(int *a, int *b, int *c, int *d) {
  *d = threadIdx.x;

  *c = *a + *b;
}

int main(void) {
  int a, b, c=0, d;
  int *da, *db, *dc, *dd;

  cudaMalloc(&da, sizeof(int));
  cudaMalloc(&db, sizeof(int));
  cudaMalloc(&dc, sizeof(int));
  cudaMalloc(&dd, sizeof(int));

  a = 2;
  b = 9;

  cudaMemcpy(da, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(db, &b, sizeof(int), cudaMemcpyHostToDevice);

  mykernel<<<1,2>>>(da, db, dc, dd);

  printf("Hello World! %d %d %d %d\n", a, b, c, d);

  cudaMemcpy(&c, dc, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&d, dd, sizeof(int), cudaMemcpyDeviceToHost);

  printf("Hello World! %d %d %d %d\n", a, b, c, d);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dd);

  return 0;
}
