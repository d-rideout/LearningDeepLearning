F = mpif90

nn: nn.h nn.cu
	time nvcc -g -G -o nn --compiler-options -Wall nn.cu

dl: dl.f95
	$F -cpp dl.f95
