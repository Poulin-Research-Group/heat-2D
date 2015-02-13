      subroutine heatf(un,u,Nx,Ny,C,Kx,Ky)
      implicit none

      double precision un(Nx,Ny),u(Nx,Ny)
      double precision C,Kx,Ky
        
      integer Nx,Ny,i,j
cf2py intent(in) :: C,Kx,Ky
cf2py intent(in) :: u
cf2py intent(in,out) :: un
cf2py intent(hide) :: Nx,Ny

      ! print *, C, Kx, Ky


      do j=2,Nx-1
        do i=2,Ny-1
          un(i,j)=C*u(i,j)+Kx*(u(i+1,j)+u(i-1,j))+Ky*(u(i,j+1)+u(i,j-1))
        enddo
      enddo

      end
