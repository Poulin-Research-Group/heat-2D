      subroutine heatf(u, Nx, Ny, K, Kx, Ky)
      implicit none

      double precision u(0:Ny+1, 0:Nx+1)
      double precision K, Kx, Ky
      integer Nx, Ny
        
cf2py intent(in) :: K, Kx, Ky
cf2py intent(hide) :: Nx, Ny
cf2py intent(in,out) :: u

      u(1:Ny, 1:Nx) = K*u(1:Ny,1:Nx)
     &              + Ky*(u(2:Ny+1, 1:Nx) + u(0:Ny-1, 1:Nx))
     &              + Kx*(u(1:Ny, 2:Nx+1) + u(1:Ny, 0:Nx-1))

      end
