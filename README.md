# heat-2d
solving the heat equation using python, fortran77 and fortran90

## running the code
before the code is run, the Fortran code must be compiled using f2py.

### compiling

```
f2py -c -m heatFortran77 heatFortran.f
f2py -c -m heatFortran90 heatFortran.f90
```

gfortran optimization options can be used, e.g.

```
f2py -c -m --opt=-O3 heatFortran77 heatFortran.f
f2py -c -m --opt=-O3 heatFortran90 heatFortran.f90
```

to use the `Ofast` optimization, it is possible that the stack limit needs to be changed. to check what your stack limit is, run

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

### actually running the code
edit the different parameters in the `main.py` file
