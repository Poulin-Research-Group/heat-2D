# heat-2d
Solving the 2-D heat equation using Python, Fortran 77 and Fortran 90.

## Running the code
Before the code is run, the Fortran code must be compiled using f2py.

### Compiling

```
f2py -c -m heatFortran77 heatFortran.f
f2py -c -m heatFortran90 heatFortran.f90
```

gfortran optimization options can be used, e.g.

```
f2py -c -m --opt=-O3 heatFortran77 heatFortran.f
f2py -c -m --opt=-O3 heatFortran90 heatFortran.f90
```

To use the `Ofast` optimization, it is possible that the stack limit needs to be changed. To check what your stack limit is, run

```
ulimit -s
```

then, change it to unlimited and compile the code:

```
ulimit -s unlimited
f2py -c -m --opt=-Ofast heatFortran77 heatFortran.f
f2py -c -m --opt=-Ofast heatFortran90 heatFortran.f90
```

and then you can change the stack limit back to whatever it was before, using

```
ulimit -s OLD_STACK_LIMIT
```

where `OLD_STACK_LIMIT` is the value.

### Actually running the code
Edit the different parameters in the `main.py` file.


## Running tests
There are multiple parameters that can be specified when running the tests. These are:

- `-px` : the number of processors to use for the x-dimension
- `-py` : the number of processors to use for the y-dimension
- `-sc_x` : the scaling factor for the x-dimension
- `-sc_y` : the scaling factor for the y-dimension
- `-T` : the number of trials
- `-opt` : the optimization to use for f2py
- `--square` : if set, the grid is square

`-px`, `-py`, `-sc_x`, `-sc_y` are all ranges of values.

For each range of values, a loop is run over them and the Python, Fortran 77 and Fortran 90 code are run for these scenarios, e.g.

```
bash run_tests.sh -px 1,2 -py 1,1 -sc_x 4,8 -sc_y 2,4 -T 100 -opt O3
```

will run 100 trials of the three solvers, optmized with O3, for:

| p_x | p_y | sc_x | sc_y |
| --: | --: | ---: | ---: |
| 1   | 1   | 4    | 2    |
| 1   | 1   | 4    | 4    |
| 1   | 1   | 8    | 2    |
| 1   | 1   | 8    | 4    |
| 2   | 1   | 4    | 2    |
| 2   | 1   | 4    | 4    |
| 2   | 1   | 8    | 2    |
| 2   | 1   | 8    | 4    |

If the `--square` flag was set, then the `-sc_y` parameter is ignored and the values for `sc_y` are the same as `sc_x`.

**NOTE**: only powers of 2 will be run, because I said so. If you were to specify, for example, `-px 1,3` then the code would run solvers for `px = 1` and `px = 2`, i.e. the nearest power of 2.

## File Descriptions
I confused myself with my poor naming scheme. Here's some details:

- **main.py**: the parameters of the problem
- **setup.py**: all of the different solution methods
- **stepping.py**: (old version? I think)
- **run_tests.sh**: shell script to run the tests
