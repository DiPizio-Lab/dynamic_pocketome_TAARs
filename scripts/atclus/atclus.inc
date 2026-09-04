c---->---1---------2---------3---------4---------5---------6---------7--<
      implicit double precision(a-h,o-z)
c      
      parameter (maxatm=50000,maxgrp=99)
      parameter (pi=3.141592653589793d0)
      parameter ( z0= 0.d0, z1= 1.d0, z2= 2.d0, z3= 3.d0, z4= 4.d0,
     &            z5= 5.d0, z6= 6.d0, z7= 7.d0, z8= 8.d0, z9= 9.d0,
     &           z10=10.d0,z11=11.d0,z12=12.d0,z13=13.d0,z14=14.d0,
     &         z50=50.d0,z100=100.d0,z180=180.d0 )
      parameter ( f2=0.5d0,f4=0.25d0,f5=0.2d0,f12=0.08333333333333333d0)
      parameter ( tol=1.d-3,wuzeps=1.d-8,bignum=1.d100 )
      parameter ( tolaus=1.d-15)
c
      logical logfil
      character*80 chalin,chafil,charad,chaout,chaarg(2),chalog
c
      common /charac/ chalin,chafil,charad,chaout,chaarg,chalog
      common /logicl/ logfil 
      common /intfil/ ianfil,ienfil,lchfil,ianlog,ienlog,lchlog
      common /intsca/ numatm,numele,neimax,numgrp
      common /dblsca/ radcon,dismin,dismax,dsmiel,dsmael
      common /intvec/ knoele(2,maxatm),ibndat(maxatm),namgrp(maxgrp),
     &                igroup(maxatm),isrgrp(maxatm),nuprgr(maxgrp),
     &                namold(maxgrp)
      common /dblvec/ xyzatm(3,maxatm),xyzcog(3,maxgrp)
c
c---->---1---------2---------3---------4---------5---------6---------7--<
