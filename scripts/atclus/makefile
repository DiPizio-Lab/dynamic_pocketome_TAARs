#
#	Makefile fuer *** atclus.f ***
#


FFLAGS = -g -C -w
FFLAGO = -O3 -w


#
#	Objectfiles fuer atclus
#


atclusobjs =	atclus.o 
atclusdbx  =    atclus_dbx.o
atcluslinks =  

#
#	linken 
#
all:	atclus atclus_dbx

atclus_dbx: ${atclusdbx} 
	@echo "    gfortran -o atclus_dbx ..."
	@gfortran ${FFLAGO} -o atclus_dbx ${atclusdbx} ${atcluslinks}

atclus: ${atclusobjs} 
	@echo "    gfortran -o atclus ..."
	@gfortran ${FFLAGS} -o atclus ${atclusobjs} ${atcluslinks}



#
#	Source code definition for compiling
#


atclus.o:	atclus.f atclus.inc
		gfortran $(FFLAGO) -c atclus.f

atclus_dbx.o:	atclus.f atclus.inc
		gfortran $(FFLAGS) -c atclus.f -o atclus_dbx.o
#
#	putzen
#


clean:
	rm -f *.o atclus atclus_dbx














