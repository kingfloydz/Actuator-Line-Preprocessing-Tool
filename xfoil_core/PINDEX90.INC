
!
!---- Pointers for referencing polar force coefficients
!     First 4 pointers must be main polar plot variables.
!
      integer, parameter :: IAL  = 1   ! alpha
      integer, parameter :: ICL  = 2   ! CL
      integer, parameter :: ICD  = 3   ! CD
      integer, parameter :: ICM  = 4   ! Cm
      integer, parameter :: ICW  = 5   ! CDwave
      integer, parameter :: ICV  = 6   ! CDvisc
      integer, parameter :: ICP  = 7   ! CDpres
      integer, parameter :: IMA  = 8   ! Mach
      integer, parameter :: IRE  = 9   ! Re
      integer, parameter :: INC  = 10  ! Ncrit
      integer, parameter :: ICH  = 11  ! Hinge moment
      integer, parameter :: IMC  = 12  ! Minimum Cp on surface
      integer, parameter :: IMCX = 13  ! X/C for minimum Cp on surface
      integer, parameter :: IPTOT = 13
!
!
!---------------------
!  Pointers for referencing polar airfoil-side quantities
!
      integer, parameter :: JTP = 1    ! trip
      integer, parameter :: JTN = 2    ! transition
      integer, parameter :: JPTOT = 2

      character(len=10) :: CPOLNAME(IPTOT)
      character(len=5)  :: CPOLSNAME(JPTOT)
      character(len=6)  :: CPOLFORM(IPTOT), CPOLSFORM(JPTOT)
!
      data CPOLNAME  / &
     & 'alpha     ', &
     & 'CL        ', &
     & 'CD        ', &
     & 'CM        ', &
     & 'CDw       ', &
     & 'CDv       ', &
     & 'CDp       ', &
     & 'Mach      ', &
     & 'Re        ', &
     & 'Ncrit     ', &
     & 'Chinge    ', &
     & 'Cpmin     ', &
     & 'XCpmin    '  /
      data CPOLFORM  / &
     & 'F7.3  ',   & ! alpha
     & 'F9.4  ',   & ! CL
     & 'F10.5 ',   & ! CD
     & 'F9.4  ',   & ! CM
     & 'F10.5 ',   & ! CDw
     & 'F10.5 ',   & ! CDv
     & 'F10.5 ',   & ! CDp
     & 'F8.4  ',   & ! Mach
     & 'E11.3 ',   & ! Re
     & 'F7.3  ',   & ! Ncrit
     & 'F9.5  ',   & ! Chinge
     & 'F9.4  ',   & ! Cpmin
     & 'F9.4  '  /   ! XCpmin

      data CPOLSNAME / 'Xtrip', 'Xtr  ' /
      data CPOLSFORM / 'F9.4  ', 'F9.4  ' /
