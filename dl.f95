!#define MPI
#define ETA .1

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
real, dimension(3) :: a
real :: b, z !, eta=.1

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
  call random_seed(size = i)
  print *, 'seed size:', i
  call random_seed(put=[(0, j=1, i)])

  call random_number(a)
  call random_number(b)
  !if (myproc==0) print *, 'orig:', a, b
  a = 2*a-1
  b = 2*b-1
  if (myproc==0) print *, '          a =', a, 'b =', b

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

     print *
     print *, i, ':', in

     ! Compute output on data
     z = sum(a*in+b)
     print *, i, ':', a*in+b, ':', z

     ! Learn?
     if (z>0) then
        print *, "prime"
        if (y(i)==1) then
           print *, 'correct!'
        else
           call learn(a, b, in, -1.)
        end if
     else
        print *, "composite"
        if (y(i)==0) then
           print *, 'correct!'
        else 
           call learn(a, b, in, 1.)
        end if
     end if

  end do
  !in = [( iand(i+2, shiftl(1,i)), i=0,5)]
  !print *, 'in =', in


  
#ifdef MPI
  call MPI_FINALIZE(ierr)
#endif
#endif
end program dl


subroutine learn(a, b, in, sign)
real, dimension(3), intent(inout) :: a
real, intent(inout) :: b
real, dimension(3), intent(in) :: in
real, intent(in) :: sign

a = a - ETA*in
b = b + ETA*sign
print *, 'learn'
print *, ETA, a, b

end subroutine learn
