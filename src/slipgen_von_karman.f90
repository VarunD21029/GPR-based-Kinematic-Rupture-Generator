! ================================================================
! slipgen_vk.f90
! Spatial Correlation Slip Generator
! Supports: Von Kármán, Exponential, and Gaussian filters
!
! INPUT FILE — slipgen.in:
!   Line 1:  L    W         fault length, width (km)
!   Line 2:  M    N         FFT grid points
!   Line 3:  ax   az        correlation lengths (km)
!   Line 4:  itype  H       itype (1=VK, 2=Exp, 3=Gauss), Hurst (H)
!   Line 5:  NX   NY        subfault grid
!   Lines 6+: subfault slip values (NY lines, NX values each)
! ================================================================

PROGRAM slipgen_vk
    IMPLICIT NONE
    REAL*8, PARAMETER :: PI = 3.14159265358979323846D0

    ! Original variables
    INTEGER           :: i, j, k, NX, NY, NXX, NYY, M, N, FNN, FNM
    REAL*8            :: dkx, dky, L, W, kx, ky, dx, dy
    REAL*8            :: cx, cy, ms, NyqL, NyqW, NLW
    REAL*8            :: krad
    INTEGER           :: ci, cj, lL, rL, bW, tW, dw

    ! Spatial Correlation variables
    INTEGER           :: itype           ! 1=VK, 2=Exp, 3=Gauss
    REAL*8            :: ax, az          ! correlation lengths (km)
    REAL*8            :: H_vk            ! Hurst exponent
    REAL*8            :: ax_sc, az_sc    ! ax*2pi, az*2pi
    REAL*8            :: vk_exp          ! -(H_vk + 1)
    REAL*8            :: K2_rad, K2_cyc  ! radial wavenumber squared
    REAL*8            :: P_val           ! Power at (kx,ky)
    REAL*8            :: amp_dc          ! ABS(AC(1,1))

    COMPLEX*16, ALLOCATABLE :: speq1(:), AC(:,:)
    COMPLEX*16, ALLOCATABLE :: speqd(:), DC(:,:)
    REAL*8,    ALLOCATABLE  :: A(:,:), AA(:,:), C(:,:), D(:,:)
    COMPLEX*16, ALLOCATABLE :: A_cplx(:,:), D_cplx(:,:)

    OPEN(101, FILE='slipgen.txt')
    OPEN(105, FILE='specx.txt')
    OPEN(106, FILE='specy.txt')
    OPEN(130, FILE='slipgen.in')
    OPEN(120, FILE='slipx.txt')
    OPEN(121, FILE='slipy.txt')

    CALL RANDOM_SEED()

    !----------------------------------------------------------
    ! Read input 
    !----------------------------------------------------------
    READ(130,*) L, W
    READ(130,*) M, N
    READ(130,*) ax, az       
    READ(130,*) itype, H_vk  ! NEW: Model type and Hurst

    ! Pre-compute VK constants
    ax_sc  = ax * 2.0D0 * PI       
    az_sc  = az * 2.0D0 * PI
    vk_exp = -(H_vk + 1.0D0)       

    FNN = N/2 + 1
    FNM = M/2 + 1

    ALLOCATE(speq1(N), speqd(N))
    ALLOCATE(AC(M/2,N), DC(M/2,N))
    ALLOCATE(A(M,N), AA(M,N), C(M,N), D(M,N))
    ALLOCATE(A_cplx(M/2,N), D_cplx(M/2,N))

    dkx = 1.0D0 / L
    dky = 1.0D0 / W
    dx  = L / REAL(M)
    dy  = W / REAL(N)

    ! Console Output
    WRITE(*,'(A)') '=== Spatial Correlation Slip Generator ==='
    IF (itype == 1) THEN
        WRITE(*,'(A)') '  Model: Von Karman'
    ELSE IF (itype == 2) THEN
        WRITE(*,'(A)') '  Model: Exponential'
    ELSE IF (itype == 3) THEN
        WRITE(*,'(A)') '  Model: Gaussian'
    ELSE
        WRITE(*,'(A)') '  ERROR: Unknown itype. Defaulting to Von Karman.'
        itype = 1
    END IF
    
    WRITE(*,'(A,F7.2,A,F7.2)') '  L=',L,' km  W=',W,' km'
    WRITE(*,'(A,F7.3,A,F7.3,A)') '  ax=',ax,' km  az=',az,' km'
    IF (itype == 1) WRITE(*,'(A,F7.3)') '  Hurst=',H_vk
    WRITE(*,'(A,F8.4,A,F8.4)') '  dx=',dx,' km  dy=',dy,' km'

    !----------------------------------------------------------
    ! White noise phase generation
    !----------------------------------------------------------
    DO i = 1, M
        DO j = 1, N
            CALL RANDOM_NUMBER(D(i,j))
        END DO
    END DO

    DO i = 1, M/2
        D_cplx(i,:) = CMPLX(D(2*i-1,:), D(2*i,:))
    END DO
    CALL rlft3(D_cplx, speqd, M, N, 1, 1)

    speqd = EXP(CMPLX(0.D0, ATAN2(AIMAG(speqd), DBLE(REAL(speqd)))))
    DO i = 1, M/2
        DC(i,:) = EXP(CMPLX(0.D0, ATAN2(AIMAG(D_cplx(i,:)), &
                                          DBLE(REAL(D_cplx(i,:))))))
    END DO

    !----------------------------------------------------------
    ! Subfault setup 
    !----------------------------------------------------------
    READ(130,*) NX, NY
    NyqL = REAL(NX) / L
    NyqW = REAL(NY) / W
    NLW  = SQRT(NyqL**2 + NyqW**2)

    DO j = 1, NY
        READ(130,*) (C(i,j), i=1,NX)
    END DO

    NXX = M / NX
    NYY = N / NY
    ms  = SUM(C(1:NX,1:NY)) / NX / NY

    DO j = 1, N
        DO i = 1, M
            AA(i,j) = C((i-1)/NXX+1, (j-1)/NYY+1)
        END DO
    END DO

    !----------------------------------------------------------
    ! Smoothing 
    !----------------------------------------------------------
    dw = 2
    DO i = 1, M
        DO j = 1, N
            lL = MAX(1, i-dw)
            rL = MIN(M, i+dw)
            bW = MAX(1, j-dw)
            tW = MIN(N, j+dw)
            A(i,j) = SUM(AA(lL:rL, bW:tW)) / (dw*2+1)**2
        END DO
    END DO

    !----------------------------------------------------------
    ! Forward FFT
    !----------------------------------------------------------
    DO i = 1, M/2
        A_cplx(i,:) = CMPLX(A(2*i-1,:), A(2*i,:))
    END DO
    CALL rlft3(A_cplx, speq1, M, N, 1, 1)
    DO i = 1, M/2
        AC(i,:) = A_cplx(i,:)
    END DO

    amp_dc = ABS(AC(1,1))

    !----------------------------------------------------------
    ! Spatial Spectral Filter Applied
    !----------------------------------------------------------
    DO j = 1, N
        IF (j <= N/2+1) THEN
            ky = dky * REAL(j-1)
        ELSE
            ky = -dky * REAL(N-j+1)
        END IF

        DO i = 1, M/2+1
            kx   = dkx * REAL(i-1)
            krad = SQRT(kx**2 + ky**2)

            IF (krad >= NLW) THEN
                ! Select the appropriate power spectrum shape
                IF (itype == 1) THEN         ! Von Karman
                    K2_rad = (ax_sc*kx)**2 + (az_sc*ky)**2
                    P_val = (1.0D0 + K2_rad)**vk_exp
                
                ELSE IF (itype == 2) THEN    ! Exponential (VK with H=0.5)
                    K2_rad = (ax_sc*kx)**2 + (az_sc*ky)**2
                    P_val = (1.0D0 + K2_rad)**(-1.5D0)
                
                ELSE                         ! Gaussian
                    K2_cyc = (ax*kx)**2 + (az*ky)**2
                    P_val  = EXP(-(PI**2) * K2_cyc)
                END IF

                ! Apply amplitude filter (SQRT of Power)
                IF (i < M/2+1) THEN
                    AC(i,j) = amp_dc * SQRT(P_val) * DC(i,j)
                ELSE
                    speq1(j) = amp_dc * SQRT(P_val) * speqd(j)
                END IF
            END IF
        END DO
    END DO

    !----------------------------------------------------------
    ! Inverse FFT 
    !----------------------------------------------------------
    DO i = 1, M/2
        A_cplx(i,:) = AC(i,:)
    END DO
    CALL rlft3(A_cplx, speq1, M, N, 1, -1)
    DO i = 1, M/2
        A(2*i-1,:) = DBLE(REAL(A_cplx(i,:))) / M / N * 2.D0
        A(2*i,  :) = AIMAG(A_cplx(i,:))      / M / N * 2.D0
    END DO

    !----------------------------------------------------------
    ! Cut negatives 
    !----------------------------------------------------------
    A = EXP(A / MAXVAL(ABS(A)))

    !----------------------------------------------------------
    ! Cosine taper 
    !----------------------------------------------------------
    cy = W * 3/ 20.D0
    cx = L * 3/ 20.D0
    ci = INT(cx / dx)
    cj = INT(cy / dy)

    DO j = 1, cj+1
        A(:,j) = A(:,j) * (0.5D0 + 0.5D0*COS(PI*REAL(cj-j+1)/REAL(cj)))
        k = N - cj + j - 1
        A(:,k) = A(:,k) * (0.5D0 + 0.5D0*COS(PI*REAL(k-N+cj)/REAL(cj)))
    END DO
    DO i = 1, ci+1
        A(i,:) = A(i,:) * (0.5D0 + 0.5D0*COS(PI*REAL(ci-i+1)/REAL(ci)))
        k = M - ci + i - 1
        A(k,:) = A(k,:) * (0.5D0 + 0.5D0*COS(PI*REAL(k-M+ci)/REAL(ci)))
    END DO

    !----------------------------------------------------------
    ! Impose mean slip 
    !----------------------------------------------------------
    WRITE(*,'(A,F10.5)') '  Mean slip target: ', ms
    A = A * ms / SUM(A) * REAL(M*N)
    WRITE(*,'(A,F10.5)') '  Mean slip actual: ', SUM(A)/REAL(M*N)

    !----------------------------------------------------------
    ! Write slipgen.txt 
    !----------------------------------------------------------
    DO i = 1, M
        DO j = 1, N
            WRITE(101,'(3E13.6)') REAL(i-1)*dx, REAL(j-1)*dy, A(i,j)
        END DO
    END DO

    j = N/2
    DO i = 1, M
        WRITE(120,*) REAL(i-1)*dx, A(i,j)
    END DO

    i = M/2
    DO j = 1, N
        WRITE(121,*) REAL(j-1)*dy, A(i,j)
    END DO

    !----------------------------------------------------------
    ! Spectrum output 
    !----------------------------------------------------------
    DO i = 1, M/2
        A_cplx(i,:) = CMPLX(A(2*i-1,:), A(2*i,:))
    END DO
    CALL rlft3(A_cplx, speq1, M, N, 1, 1)
    DO i = 1, M/2
        AC(i,:) = A_cplx(i,:)
    END DO
    AC    = AC    / REAL(M*N/2)
    speq1 = speq1 / REAL(M*N/2)

    DO i = 1, N/2+1
        WRITE(106,*) (i-1)*dky, ABS(AC(1,i))
    END DO
    DO i = 1, M/2
        WRITE(105,*) (i-1)*dkx, ABS(AC(i,1))
    END DO
    WRITE(105,*) (M/2)*dkx, ABS(speq1(1))

    DEALLOCATE(speq1,speqd,AC,DC,A,AA,C,D,A_cplx,D_cplx)

END PROGRAM slipgen_vk


! ================================================================
! EXAMPLE slipgen.in for Athens fault (11x8 km, Mw~5.9):
!
!   11.0    8.0         ! L (km), W (km)
!   256     256         ! M, N
!   3.85    2.80        ! ax (km), az (km) 
!   1       0.70        ! Model (1=VK, 2=Exp, 3=Gauss) & H (Hurst)
!   3       3           ! NX, NY
!   0.18    0.18   0.18
!   0.18    0.54   0.18
!   0.18    0.18   0.18
! ================================================================

!==============================================================
! SUBROUTINE rlft3 — UNCHANGED
!==============================================================
SUBROUTINE rlft3(data, speq, nn1, nn2, nn3, isign)
    IMPLICIT NONE
    INTEGER,    INTENT(IN)    :: isign, nn1, nn2, nn3
    COMPLEX*16, INTENT(INOUT) :: data(nn1/2, nn2, nn3)
    COMPLEX*16, INTENT(INOUT) :: speq(nn2, nn3)

    INTEGER    :: i1, i2, i3, j1, j2, j3, nn(3)
    REAL*8     :: theta, wi, wpi, wpr, wr, wtemp
    COMPLEX*16 :: c1, c2, h1, h2, w

    c1    = DCMPLX(0.5D0,  0.0D0)
    c2    = DCMPLX(0.0D0, -0.5D0*isign)
    theta = 6.28318530717959D0 / DBLE(isign*nn1)
    wpr   = -2.0D0 * SIN(0.5D0*theta)**2
    wpi   =  SIN(theta)
    nn(1) = nn1/2
    nn(2) = nn2
    nn(3) = nn3

    IF (isign == 1) THEN
        CALL fourn(data, nn, 3, isign)
        DO i3 = 1, nn3
            DO i2 = 1, nn2
                speq(i2,i3) = data(1,i2,i3)
            END DO
        END DO
    END IF

    DO i3 = 1, nn3
        j3 = 1
        IF (i3 /= 1) j3 = nn3-i3+2
        wr = 1.0D0
        wi = 0.0D0
        DO i1 = 1, nn1/4+1
            j1 = nn1/2-i1+2
            DO i2 = 1, nn2
                j2 = 1
                IF (i2 /= 1) j2 = nn2-i2+2
                IF (i1 == 1) THEN
                    h1 = c1*(data(1,i2,i3) + CONJG(speq(j2,j3)))
                    h2 = c2*(data(1,i2,i3) - CONJG(speq(j2,j3)))
                    data(1,i2,i3) = h1 + h2
                    speq(j2,j3)   = CONJG(h1 - h2)
                ELSE
                    h1 = c1*(data(i1,i2,i3) + CONJG(data(j1,j2,j3)))
                    h2 = c2*(data(i1,i2,i3) - CONJG(data(j1,j2,j3)))
                    data(i1,i2,i3) = h1 + w*h2
                    data(j1,j2,j3) = CONJG(h1 - w*h2)
                END IF
            END DO
            wtemp = wr
            wr    = wr*wpr - wi*wpi + wr
            wi    = wi*wpr + wtemp*wpi + wi
            w     = DCMPLX(wr, wi)
        END DO
    END DO

    IF (isign == -1) THEN
        CALL fourn(data, nn, 3, isign)
    END IF

END SUBROUTINE rlft3


!==============================================================
! SUBROUTINE fourn — UNCHANGED
!==============================================================
SUBROUTINE fourn(data, nn, ndim, isign)
    IMPLICIT NONE
    INTEGER,    INTENT(IN)    :: isign, ndim, nn(ndim)
    COMPLEX*16, INTENT(INOUT) :: data(*)

    INTEGER :: i1,i2,i2rev,i3,i3rev,ibit,idim,ifp1,ifp2
    INTEGER :: ip1,ip2,ip3,k1,k2,n,nprev,nrem,ntot
    REAL*8  :: tempi, tempr, theta, wi, wpi, wpr, wr, wtemp
    COMPLEX*16 :: temp

    ntot = 1
    DO idim = 1, ndim
        ntot = ntot * nn(idim)
    END DO

    nprev = 1
    DO idim = 1, ndim
        n    = nn(idim)
        nrem = ntot / (n*nprev)
        ip1  = nprev
        ip2  = ip1*n
        ip3  = ip2*nrem
        i2rev = 1

        DO i2 = 1, ip2, ip1
            IF (i2 < i2rev) THEN
                DO i1 = i2, i2+ip1-1
                    DO i3 = i1, ip3, ip2
                        i3rev       = i2rev+i3-i2
                        temp        = data(i3)
                        data(i3)    = data(i3rev)
                        data(i3rev) = temp
                    END DO
                END DO
            END IF
            ibit = ip2/2
            DO WHILE ((ibit >= ip1) .AND. (i2rev > ibit))
                i2rev = i2rev - ibit
                ibit  = ibit/2
            END DO
            i2rev = i2rev + ibit
        END DO

        ifp1 = ip1
        DO WHILE (ifp1 < ip2)
            ifp2  = 2*ifp1
            theta = isign * 6.28318530717959D0 / (ifp2/ip1)
            wpr   = -2.D0 * SIN(0.5D0*theta)**2
            wpi   =  SIN(theta)
            wr    = 1.D0
            wi    = 0.D0
            DO i3 = 1, ifp1, ip1
                DO i1 = i3, i3+ip1-1
                    DO i2 = i1, ip3, ifp2
                        k1    = i2
                        k2    = k1 + ifp1
                        tempr = wr*DBLE(REAL(data(k2))) - wi*AIMAG(data(k2))
                        tempi = wr*AIMAG(data(k2))      + wi*DBLE(REAL(data(k2)))
                        data(k2) = data(k1) - DCMPLX(tempr,tempi)
                        data(k1) = data(k1) + DCMPLX(tempr,tempi)
                    END DO
                END DO
                wtemp = wr
                wr    = wr*wpr - wi*wpi + wr
                wi    = wi*wpr + wtemp*wpi + wi
            END DO
            ifp1 = ifp2
        END DO
        nprev = n * nprev
    END DO

END SUBROUTINE fourn
