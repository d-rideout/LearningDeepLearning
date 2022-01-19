F = mpif90

nn: nn.h nn.cu
	time nvcc -o nn nn.cu

dl: dl.f95
	$F -cpp dl.f95
