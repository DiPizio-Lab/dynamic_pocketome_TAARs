# wrapping with vmd
mol load psf NEUTRAL_fis.psf dcd output.dcd
package require pbctools
set [pbc get -all] -all
pbc wrap -center bb -centersel "protein and segid P0" -compound res -all
animate write dcd {./output_wrapped.dcd} beg 0 end -1 skip 1 0
exit
