# make test directory if it doesn't exist
if [ ! -d "tests" ]; then
  echo Creating test directories...
  mkdir -p tests/numpy tests/f2py-f77 tests/f2py-f90
fi

function test_python () {
  sc="$1"
  T="$2"
  px="$3"
  py="$4"
  p=$(expr $px \* $py)

  echo numpy
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python stepping_mpi_2D.py numpy $sc $px $py
  done
}


function test_f2py77 () {
  sc="$1"
  T="$2"
  px="$3"
  py="$4"
  p=$(expr $px \* $py)

  echo f2py-f77
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python stepping_mpi_2D.py f2py-f77 $sc $px $py
  done
}


function test_f2py90 () {
  sc="$1"
  T="$2"
  px="$3"
  py="$4"
  p=$(expr $px \* $py)

  echo f2py-f90
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python stepping_mpi_2D.py f2py-f90 $sc $px $py
  done
}


function test_all () {
  sc="$1"
  T="$2"
  px="$3"
  py="$4"

  echo px = $px, py = $py
  test_python $sc $T $px $py
  test_f2py77 $sc $T $px $py
  test_f2py90 $sc $T $px $py
  echo
}


# number of trials; change as need be.
T=1

echo Compiling Fortran code with Ofast optimization...
f2py --opt=-Ofast -c -m heatFortran   heatFortran.f 2>/dev/null 1>&2
f2py --opt=-Ofast -c -m heatFortran90 heatFortran.f90 2>/dev/null 1>&2
echo Compiled.

for sc in 1 2; do
  echo sc = $sc, $T trials
  echo

  test_all $sc $T 1 1   # serial
  test_all $sc $T 2 1   # px = 2, py = 1
  test_all $sc $T 1 2   # px = 1, py = 2
  test_all $sc $T 2 2   # px = 2, py = 2
  test_all $sc $T 4 1
  test_all $sc $T 1 4
done