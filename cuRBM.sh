#!/bin/bash
#for v in 1024 2048 4096 8192 32768 65536 131072 262144 524288 1048576
for v in 32768 
do
  for h in 1024	2048 4096 8192 16384 32768
    do
      #./concStrmRBM 192 192 $v $h 256 $(( h/16 )) 4
      ./cuRBM 192 192 $v $h 256 $(( h/16 )) 4
    done
done
