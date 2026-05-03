module xfoil_core_bindings
  use iso_c_binding
contains
    subroutine xfoil_compute_cl_cd(npts, x_coords, y_coords, &
      n_pos, alpha_pos, n_neg, alpha_neg, &
      reynolds, mach, iter_limit, n_panels, cv_par, cte_ratio, ctr_ratio, &
      cl_pos, cd_pos, status_pos, cl_neg, cd_neg, status_neg) &
      bind(C, name="xfoil_compute_cl_cd")
    use iso_c_binding
    integer(c_int), value :: npts
    integer(c_int), value :: n_pos
    real(c_double), intent(in) :: alpha_pos(*)
    integer(c_int), value :: n_neg
    real(c_double), intent(in) :: alpha_neg(*)
    integer(c_int), value :: iter_limit
    integer(c_int), value :: n_panels
    real(c_double), value :: cv_par
    real(c_double), value :: cte_ratio
    real(c_double), value :: ctr_ratio
    real(c_double), intent(in) :: x_coords(*), y_coords(*)
    real(c_double), value :: reynolds
    real(c_double), value :: mach
    real(c_double), intent(out) :: cl_pos(*)
    real(c_double), intent(out) :: cd_pos(*)
    integer(c_int), intent(out) :: status_pos(*)
    real(c_double), intent(out) :: cl_neg(*)
    real(c_double), intent(out) :: cd_neg(*)
    integer(c_int), intent(out) :: status_neg(*)

    integer :: i, ip, idx
    integer :: max_iter
    real(c_double) :: area, alpha
    real(c_double) :: cl_val, cd_val

    include 'XFOIL90.INC'

    do idx = 1, n_pos
      cl_pos(idx) = 0.0_c_double
      cd_pos(idx) = 0.0_c_double
      status_pos(idx) = 0
    end do
    do idx = 1, n_neg
      cl_neg(idx) = 0.0_c_double
      cd_neg(idx) = 0.0_c_double
      status_neg(idx) = 0
    end do

    if (npts < 3 .or. npts > IBX) then
      return
    end if

    call init()

    LVISC = .TRUE.
    LBLINI = .FALSE.
    LPACC = .FALSE.
    LWAKE = .FALSE.
    LIPAN = .FALSE.
    LVCONV = .FALSE.
    LSCINI = .FALSE.
    LQSPEC = .FALSE.
    LCPREF = .FALSE.
    LQAIJ = .FALSE.
    LGAMU = .FALSE.
    LFOREF = .FALSE.
    LPLIST = .FALSE.

    MATYP = 1
    RETYP = 1

    REINF1 = real(reynolds, kind=KIND(REINF1))
    MINF1  = real(mach,     kind=KIND(MINF1))

    NAME = 'PYXFOIL'
    CALL STRIP(NAME,NNAME)

    NB = npts
    area = 0.0_c_double
    do i = 1, npts
      ip = i + 1
      if (ip > npts) ip = 1
      area = area + 0.5_c_double * (y_coords(i) + y_coords(ip)) * (x_coords(i) - x_coords(ip))
    end do

    if (area < 0.0_c_double) then
      LCLOCK = .TRUE.
      do i = 1, npts
        XB(i) = real(x_coords(npts - i + 1), kind=KIND(XB(1)))
        YB(i) = real(y_coords(npts - i + 1), kind=KIND(YB(1)))
      end do
    else
      LCLOCK = .FALSE.
      do i = 1, npts
        XB(i) = real(x_coords(i), kind=KIND(XB(1)))
        YB(i) = real(y_coords(i), kind=KIND(YB(1)))
      end do
    end if

    call SCALC(XB, YB, SB, NB)
    call SEGSPL(XB, XBP, SB, NB)
    call SEGSPL(YB, YBP, SB, NB)
    call GEOPAR(XB, XBP, YB, YBP, SB, NB, W1, &
          SBLE, CHORDB, AREAB, RADBLE, ANGBTE, &
          EI11BA, EI22BA, APX1BA, APX2BA, &
          EI11BT, EI22BT, APX1BT, APX2BT, &
          THICKB, CAMBRB)

    call ABCOPY(.FALSE.)

    if (n_panels > 0) then
      NPAN = MAX(4, MIN(int(n_panels), IQX-6))
    else
      NPAN = MAX(4, MIN(N, IQX-6))
    end if

    CVPAR = real(cv_par, kind=KIND(CVPAR))
    CTERAT = real(cte_ratio, kind=KIND(CTERAT))
    CTRRAT = real(ctr_ratio, kind=KIND(CTRRAT))

    call PANGEN(.FALSE.)

    N = NPAN
    NW = MAX(2, MIN(IWX, N/8 + 2))

    call SCALC(X, Y, S, N)
    call SEGSPL(X, XP, S, N)
    call SEGSPL(Y, YP, S, N)
    call NCALC(X, Y, S, N, NX, NY)

    call MRCL(1.0, MINF_CL, REINF_CL)
    call COMSET

    max_iter = MAX(1, iter_limit)
    ITMAX = max_iter

    ! --- Positive Sequence ---
    LBLINI = .FALSE.
    LWAKE = .FALSE.
    LIPAN = .FALSE.

    do idx = 1, n_pos
      alpha = alpha_pos(idx)
      ADEG = real(alpha, kind=KIND(ADEG))
      ALFA = ADEG * DTOR
      LALFA = .TRUE.
      LVCONV = .FALSE.

      call SPECAL
      call VISCAL(max_iter)

      cl_val = real(CL, kind=KIND(cl_pos(1)))
      cd_val = real(CD, kind=KIND(cd_pos(1)))

      if (LVCONV) then
         cl_pos(idx) = cl_val
         cd_pos(idx) = cd_val
         status_pos(idx) = 1
      else
         cl_pos(idx) = 0.0_c_double
         cd_pos(idx) = 0.0_c_double
         status_pos(idx) = 0
      end if
    end do

    ! --- Negative Sequence ---
    LBLINI = .FALSE.
    LWAKE = .FALSE.
    LIPAN = .FALSE.

    do idx = 1, n_neg
      alpha = alpha_neg(idx)
      ADEG = real(alpha, kind=KIND(ADEG))
      ALFA = ADEG * DTOR
      LALFA = .TRUE.
      LVCONV = .FALSE.

      call SPECAL
      call VISCAL(max_iter)

      cl_val = real(CL, kind=KIND(cl_neg(1)))
      cd_val = real(CD, kind=KIND(cd_neg(1)))

      if (LVCONV) then
         cl_neg(idx) = cl_val
         cd_neg(idx) = cd_val
         status_neg(idx) = 1
      else
         cl_neg(idx) = 0.0_c_double
         cd_neg(idx) = 0.0_c_double
         status_neg(idx) = 0
      end if
    end do
  end subroutine xfoil_compute_cl_cd
end module xfoil_core_bindings
