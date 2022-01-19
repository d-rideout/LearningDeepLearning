//#include <stdio.h>
//#define N 4

__global__ void mykernel(int *a, int *b, int *c) {
  int i = threadIdx.x;

  c[i] = a[i] + b[i];
}


int main(void) {
  int a[N], b[N], sum[N];
  int *da, *db, *dsum;
  size_t size = N*sizeof(int);

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

  return 0;
}
