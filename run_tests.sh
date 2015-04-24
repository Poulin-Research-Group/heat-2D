# make test directory if it doesn't exist
if [ ! -d "tests" ]; then
  echo Creating test directories...
  mkdir -p tests/numpy tests/f2py77 tests/f2py90
fi

# retrieve old stack limit, set new stack limit to unlimited
old_stack_lim=$(ulimit -s)
echo Current stack limit is $old_stack_lim. Setting it to unlimited.
ulimit -s unlimited

function test_python () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"
  p=$(expr $px \* $py)

  echo numpy
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python stepping_mpi_2D.py numpy $px $py $sc_x $sc_y
  done
  echo
}


function test_f2py77 () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"
  p=$(expr $px \* $py)

  echo f2py77
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python stepping_mpi_2D.py f2py77 $px $py $sc_x $sc_y
  done
  echo
}


function test_f2py90 () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"
  p=$(expr $px \* $py)

  echo f2py90
  for ((i=0; i<$T; i++)) do
    mpirun -np $p python stepping_mpi_2D.py f2py90 $px $py $sc_x $sc_y
  done
  echo
}


function test_all () {
  px="$1"
  py="$2"
  sc_x="$3"
  sc_y="$4"
  T="$5"

  echo px = $px, py = $py
  echo ------------------
  test_python $px $py $sc_x $sc_y $T
  test_f2py77 $px $py $sc_x $sc_y $T
  test_f2py90 $px $py $sc_x $sc_y $T
  echo
}


# number of trials; change as need be.
T=1

echo Compiling Fortran code with Ofast optimization...
f2py --opt=-Ofast -c -m heatFortran77 heatFortran.f 2>/dev/null 1>&2
f2py --opt=-Ofast -c -m heatFortran90 heatFortran.f90 2>/dev/null 1>&2
echo Compiled.

# this is assuming that sc_x = sc_y = sc, because of laziness from me.
for sc in 1 2; do
  echo sc = $sc, $T trials
  echo

  test_all 1 1 $sc $sc $T   # serial
  test_all 2 1 $sc $sc $T   # px = 2, py = 1
  test_all 1 2 $sc $sc $T   # px = 1, py = 2
  test_all 2 2 $sc $sc $T   # px = 2, py = 2
  test_all 4 1 $sc $sc $T
  test_all 1 4 $sc $sc $T
done

ulimit -s $old_stack_lim
echo Changed stack limit back to $old_stack_lim.
