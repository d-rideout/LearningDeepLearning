!#define MPI

program dl
implicit none

#if 0
print *, shiftl(1,2), iand(3,6), iand(shiftl(1,2), 7)
#else

#ifdef MPI
include 'mpif.h'

integer :: ierr, nproc
#endif

integer :: i, j, myproc

! Ground Truth
integer, dimension(2:7) :: x = [(i, i=2,7)], y = [1,1,0,1,0,1]
real, dimension(0:2) :: in

! Neural Network
real, dimension(3) :: a, b  

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
  !if (myproc==0) print *, 'orig:', a, b
  a = 2*a-1
  b = 2*b-1
  if (myproc==0) print *, 'a=', a, 'b=', b

  do i=2, 7
     ! Compute input 
     ! OMG this is annoying!  I should precompute this.
     ! in = (iand(i, shiftl(1,j)), j=0,2) chokes for some reason
     do j=0, 2
        if (iand(i, shiftl(1,j)) > 0) then
           in(j) = 1
        else
           in(j) = 0
        endif
     end do

     print *, i, ':', in

    ! Compute output on data
     print *, a*in+b
     !sum(a*in)
  end do
  !in = [( iand(i+2, shiftl(1,i)), i=0,5)]
  !print *, 'in =', in


  
#ifdef MPI
  call MPI_FINALIZE(ierr)
#endif
#endif
end program dl

!subroutine 
