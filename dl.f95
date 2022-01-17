#define MPI

program dl
implicit none

#ifdef MPI
include 'mpif.h'

integer :: ierr, nproc
#endif

integer :: i, myproc

! Ground Truth
integer, dimension(2:7) :: x = [(i, i=2,7)], y = [1,1,0,1,0,1]

! Neural Network
real, dimension(1) :: a, b  

! MPI initialization stuff
#ifdef MPI
  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myproc, ierr)
  !print *, 'MPI: ierr =', ierr, 'nproc =', nproc
#else
  myproc = 0
#endif

  
  ! Show Ground Truth
  !print *, (i, ':', y(i), '|', i=2,7)
  if (myproc==0) then
  print *, 'x =', x
  print *, 'y =', y
  endif
  
  ! Initialize Neural Network
!  call random_seed(put=[0,0])
!  call srand(0)
  call random_number(a)
  call random_number(b)
  if (myproc==0) print *, 'orig:', a, b
  a = 2*a-1
  b = 2*b-1
  if (myproc==0) print *, ' mod:', a, b

  ! Compute output on data
  
  ! iand()

  

  
#ifdef MPI
  call MPI_FINALIZE(ierr)
#endif
end program dl

!subroutine 
