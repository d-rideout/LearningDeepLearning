F = mpif90

nn: nn.h nn.cu
	time nvcc -g -o nn nn.cu

dl: dl.f95
	$F -cpp dl.f95
