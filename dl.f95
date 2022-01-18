!#define MPI
#define ETA .2
#define MAX_TRY 1000
#define NNEURONS 1
#define NBITS 4

! input --> NNeurons --> output neuron

program dl
implicit none

#if 0
print *, shiftl(1,2), iand(3,6), iand(shiftl(1,2), 7)
#else

#ifdef MPI
include 'mpif.h'

integer :: ierr, nproc
#endif

integer :: i, j, myproc, ni, num

! Ground Truth
integer, dimension(2:15) :: x = [(i, i=2,15)], y = [1,1,0,1,0,1,0,0,0,1,0,1,0,0]
real, dimension(0:3) :: in

! Neural Network
real, dimension(NBITS,NNEURONS) :: a1
real, dimension(NNEURONS) :: a2
real, dimension(NNEURONS) :: b1, z1
real :: b2, z2

! Outcome
integer :: nwrong=1, try=0

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

  call random_number(a1)
  call random_number(a2)
  call random_number(b1)
  call random_number(b2)
  !if (myproc==0) print *, 'orig:', a, b
  a1 = 2*a1-1
  a2 = 2*a2-1
  b1 = 2*b1-1
  b2 = 2*b2-1
  if (myproc==0) print *, '          a =', a1, 'b =', b1

  ! Main loop
  do while (nwrong>0)
     if (try > MAX_TRY) exit
     nwrong = 0
     try = try + 1

     ! Loop over data
     do num=2, 15
        ! Compute input 
        ! OMG this is annoying!  I should precompute this.
        ! in = (iand(i, shiftl(1,j)), j=0,2) chokes for some reason
        do j=0, 3
           if (iand(num, shiftl(1,j)) > 0) then
              in(j) = 1
           else
              in(j) = 0
           endif
        end do

        ! Loop over layer 1 neurons
        do ni = 1, NNEURONS
           print *, 'neuron', ni
           print '(I2.2,A6,4f6.2)', num, ': in =', in
           ! print '(I2.2,A1,4f3.0)', i, ':', in
           print '(I2.2,A6,4f6.2)', num, ':  a1 =', a1(:,ni)

           ! Compute output on data
           z1(ni) = sum(a1(:,ni)*in) + b1(ni)
           print '(i2.2, a6, 4f6.2, a6, f6.2)', num, ':  z1 =', a1(:,ni)*in, ' + b1 =', z1(ni)
        end do

        ! Feed to layer 2 neuron

        
        ! Learn?
        if (sum(z1)>0) then
           print *, "prime"
           if (y(num)==1) then
              print *, 'correct!'
           else
              !call learn(a1, b1, in, -1., nwrong)
           end if
        else
           print *, "composite"
           if (y(num)==0) then
              print *, 'correct!'
           else 
              !call learn(a1, b1, in, 1., nwrong)
           end if
        end if

     end do ! loop over data

     ! Outcome
     print *, '----------------------------------------------------------------'
     print *, 'try =', try, 'num wrong =', nwrong
  
  end do ! learning iteration

  
#ifdef MPI
  call MPI_FINALIZE(ierr)
#endif
#endif
end program dl


subroutine learn(a, b, in, sign, nwrong)
!!$real, dimension(4,NNEURONS,2), intent(inout) :: a
!!$real, dimension(NNEURONS,2), intent(inout) :: b
!!$real, dimension(4), intent(in) :: in
!!$real, intent(in) :: sign
!!$integer, intent(inout) :: nwrong

!!$print *, 'learn:', sign, ETA
!!$print *, a, 'b =', b

!!$nwrong = nwrong + 1
!!$a = a + ETA*in*sign
!!$b = b + ETA*sign
!!$print *, a, 'b =', b

call exit

end subroutine learn
