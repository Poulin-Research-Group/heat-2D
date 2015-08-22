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
    mpirun -np $p python main.py numpy $px $py $sc_x $sc_y
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
    mpirun -np $p python main.py f2py77 $px $py $sc_x $sc_y
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
    mpirun -np $p python main.py f2py90 $px $py $sc_x $sc_y
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


# default parameters
T=1
opt="O3"
px_min=1
px_max=1
py_min=1
py_max=1
sc_x_min=1
sc_x_max=1
sc_y_min=1
sc_y_max=1
square=0


# accept command line arguments
while test $# -gt 0; do
  case "$1" in
    -px)
      shift
      if test $# -gt 0; then
        IFS=, read px_min px_max <<< "$1"
      fi
      shift
      ;;

    -py)
      shift
      if test $# -gt 0; then
        IFS=, read py_min py_max <<< "$1"
      fi
      shift
      ;;

    -sc_x)
      shift
      if test $# -gt 0; then
        IFS=, read sc_x_min sc_x_max <<< "$1"
      fi
      shift
      ;;

    -sc_y)
      shift
      if test $# -gt 0; then
        IFS=, read sc_y_min sc_y_max <<< "$1"
      fi
      shift
      ;;

    -T)
      shift
      if test $# -gt 0; then
        T=$1
      fi
      shift
      ;;

    -opt)
      shift
      if test $# -gt 0; then
        opt=$1
      fi
      shift
      ;;

    --square)
      shift
      square=1
      ;;

    *)
      break
      ;;
  esac
done


printf "Compiling Fortran code with %s optimization..." $opt
f2py --opt=-$opt -c -m heatFortran77 heatFortran.f 2>/dev/null 1>&2
f2py --opt=-$opt -c -m heatFortran90 heatFortran.f90 2>/dev/null 1>&2
printf " compiled.\n"

# px, py, sc_x, sc_y are always powers of 2, e.g.
#
#   -px 1,4    --> px = [1, 2, 4]
#   -sc_x 1,16 --> sc_x = [1, 2, 4, 8, 16]
#
# if a non-power-of-2 is given, the number preceding it will be the last value, e.g.
#
#   -px 1,3  -->  px = [1, 2]
#
# if the square flag is set, then sc_y_min = sc_x_min, sc_y_max = sc_x_max and only the
# values from sc_x will be used.

for ((px=$px_min; px<=$px_max; px=px*2)); do
  for ((py=$py_min; py<=$py_max; py=py*2)); do
    if [ $square -eq 0 ]; then
      for ((sc_x=$sc_x_min; sc_x<=$sc_x_max; sc_x=2*sc_x)); do
        for ((sc_y=$sc_y_min; sc_y<=$sc_y_max; sc_y=2*sc_y)); do
          echo $px $py $sc_x $sc_y $T
          test_all $px $py $sc_x $sc_y $T
        done
      done
    else
      for ((sc=$sc_x_min; sc<=$sc_x_max; sc=2*sc)); do
        echo $px $py $sc $sc $T
        test_all $px $py $sc $sc $T
      done
    fi
  done
done

ulimit -s $old_stack_lim
printf "Changed stack limit back to %s.\n" $old_stack_lim
