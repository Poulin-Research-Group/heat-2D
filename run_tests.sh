for ((M=12; M <= 12; M++)) do
  echo M: $M
  echo serial
  for ((i=0; i<10; i++)) do
    python heat_2d_stepping.py $M $M 10000 $i 1
  done

  for ((p=2; p <= 4; p+=2)) do
  	echo parallel: p = $p
  	for ((i=0; i<10; i++)) do
	  	mpirun -np $p python heat_2d_stepping_mpi.py $M $M 10000 $i 1
	  done
	done
  echo DONE $M
done

echo DONE SWEG
