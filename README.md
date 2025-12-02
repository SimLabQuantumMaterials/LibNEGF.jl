[![DOI](https://zenodo.org/badge/1103035642.svg)](https://zenodo.org/badge/latestdoi/1103035642)

# LibNEGF

## Name
LibNEGF.jl, a Julia version of libNEGF

## Description
This repository includes implementations of some of the most computationally demanding kernels appearing in the non-equilibrium Green's function (NEGF) method. Parallelization strategies are also implemented to make use of multi-core systems. See the directory `examples/` for an easy start.

## Installation and Usage
At the moment, the default is to create synthetic data to run both the tests and the benchmarks. If you want to use realistic data, do the following: get the file `built_matrices.tar.xz`, located at `/p/project1/mat4energy/ramirez1/libNEGF/matrices/3x3/`. Extract it in your system. Let us assume this is stored in `$PATH_TO_BUILT_MATRICES`. In this repository, go to the file `setup.sh`, uncomment the block necessary to create the symbolic link to the matrices and run `./setup.sh` . This will change in the future to something more appropriate e.g. making use of Git LFS. A last step is to look for the variable `whereFrom` and set it to 1 wherever you would like to use this realistic data.

There are simple examples to get started with, in the `examples/` directory. To run those:

```
./runexamples.sh
```

To run the tests:

```
./test.sh HW
```

where `HW` is one of `cpu`, `apple`, etc. To gather and format the documentation into an HTML, go to `docs/` and run from there:

```
./make.sh HW
```

where `HW` is again one of `cpu`, `apple`, etc. This then generates the file `docs/build/index.html`.

And, to run benchmarks, to go `benchmrk` and do:

```
./runbenchmrks.sh HW T
```

where `HW` is again one of `cpu`, `apple`, etc, and `T` is 0 or 1 (1 enables backend kernel timings, 0 disables them).

## Support
`g.ramirez.hidalgo@fz-juelich.de`

## Contributing
How to contribute under construction.

## Authors and acknowledgment
Christoph Conrads - FZJ

Edoardo di Napoli - FZJ

Gustavo Ramirez-Hidalgo - FZJ

## License
BSD3, see the file `LICENSE`.