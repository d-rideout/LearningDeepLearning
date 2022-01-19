#include "nn.h"
#define MPI
#define MAX_TRY 100
#ifdef MPI
#define NPROCs 2
#else
#define NPROCs 1
#endif

! input layer ==> NNeurons ==> output neuron --> conclusion

program dl
use, intrinsic :: iso_c_binding
implicit none

interface
   integer(c_int) function compute_x(foo) bind(C, 'compute_x')
     use, intrinsic :: iso_c_binding, only : c_int
     integer(c_int) :: foo
   end function compute_x
end interface

#if 0
print *, shiftl(1,2), iand(3,6), iand(shiftl(1,2), 7)
#else

#ifdef MPI
include 'mpif.h'
integer :: ierr
#endif

integer :: i, j, myproc, ni, num, nproc=1

! Ground Truth
integer, dimension(2:15) :: x = [(i, i=2,15)], y = [1,1,0,1,0,1,0,0,0,1,0,1,0,0]
real, dimension(0:3) :: in

! Neural Network
real, dimension(NBITS,NNEURONS/NPROCs) :: a1
real, dimension(NNEURONS/NPROCs) :: b1, z1
real, dimension(NNEURONS) :: a2
real :: b2, z2
real, dimension(NNEURONS) :: z1all = 0 ! just bung all to proc 0 for now

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
  if (nproc /= NPROCs) then
     print *, 'Please set NPROCs to number of processes'
     call exit
  end if
  
  ! Show Ground Truth
  !print *, (i, ':', y(i), '|', i=2,7)
  if (myproc==0) then
  print *, 'x =', x
  print *, 'y =', y
  endif

  ! Initialize Neural Network
  call random_seed(size = i)
  if (myproc==0) print *, 'seed size:', i
  call random_seed(put=[(j*(myproc+1), j=1, i)])

  call random_number(a1)
  call random_number(a2)
  call random_number(b1)
  call random_number(b2)
  !if (myproc==0) print *, 'orig:', a, b
  a1 = 2*a1-1
  a2 = 2*a2-1
  b1 = 2*b1-1
  b2 = 2*b2-1
  !if (myproc==0)
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
        do j=0, NBITS-1
           if (1==1) then ! iand(num, shiftl(1,j)) > 0) then
              in(j) = 1
           else
              in(j) = 0
           endif
        end do

        ! Loop over layer 1 neurons
        do ni = 1, NNEURONS/NPROCs

           ! Compute output on data
           z1(ni) = sum(a1(:,ni)*in) + b1(ni)

           if (myproc==0) then
              print *, 'neuron', ni
              print '(I2.2,A6,4f6.2)', num, ': in =', in
              ! print '(I2.2,A1,4f3.0)', i, ':', in
              print '(I2.2,A6,4f6.2)', num, ': a1 =', a1(:,ni)           
              print '(i2.2, a6, 4f6.2, a7, f6.2)', num, ': z1 =', a1(:,ni)*in, ' + b1 =', z1(ni)

           else
              print '(a7, i2.2, a6, 4f6.2, a7, f6.2)', 'proc 1:', num, ': z1 =', a1(:,ni)*in, ' + b1 =', z1(ni)
           end if
        end do

        ! Send results to process 0
        if (myproc>0) then
           call mpi_send(z1, NNEURONS/NPROCs, MPI_INT, 0, 0, MPI_COMM_WORLD, ierr)
        else
           z1all(1:NNEURONS/NPROCs) = z1
           call mpi_recv(z1all(NNEURONS/NPROCs+1:), NNEURONS/NPROCs, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
        endif
        if (myproc==0) print *, 'z1all =', z1all

        ! Feed to layer 2 neuron

        
        ! Learn?
        if (sum(z1)>0) then
           if (myproc==0) print *, "prime"
           if (y(num)==1) then
              if (myproc==0) print *, 'correct!'
           else
              !call learn(a1, b1, in, -1., nwrong)
           end if
        else
           if (myproc==0) print *, "composite"
           if (y(num)==0) then
              if (myproc==0) print *, 'correct!'
           else 
              !call learn(a1, b1, in, 1., nwrong)
           end if
        end if

     end do ! loop over data

     ! Outcome
     if (myproc==0) then
        print *, '----------------------------------------------------------------'
        print *, 'try =', try, 'num wrong =', nwrong
     end if
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
