      subroutine heatf(u, Nx, Ny, K, Kx, Ky)
      implicit none

      double precision u(0:Ny+1, 0:Nx+1)
      double precision K, Kx, Ky
      integer Nx, Ny, r, c
        
cf2py intent(in) :: K, Kx, Ky
cf2py intent(hide) :: Nx, Ny
cf2py intent(in,out) :: u

      do c=1,Nx
        do r=1,Ny
          u(r,c) = K*u(r,c)
     &           + Ky*(u(r+1, c) + u(r-1, c))
     &           + Kx*(u(r, c+1) + u(r, c-1))
        enddo
      enddo

      end
