# from_GULP_to_FEFF
---
1. Convert GULP's MD simulation output *.history files to the **FEFF** input files, **PWscf** input, **XYZ**, **XSF** and for **STEM** files.
1. Calculate the average **_chi(k)_** spectrum [average from all snapshots]
1. Calculate the average **_xyz_** coordinates [average from all snapshots]
1. Calculate the **RDF** - radial distribution function [from all snapshots]
***

### Compare theory spectra with experimental data set:
___
1. Calculate the linear compositions in **_k_** and/or **_r_** spaces between serial snapshots [from all snapshots]
1. Calculate the **R**-factros (in **_k_** and **_r_** spaces)
1. Find the minimum **R**-factor snapshots composition (minimizing a multivariable function by using **BFGS** or **Differential evolution** methods)
1. Compare 2 and 3 models with an experiment and find the best linear combination between these models. (Ex. model **A** - _monomer_, model **B** - _dimer_, and best fit result is: **0.75** * _monomer_ + **0.25** *  _dimer_)
1. Calculation procedures have serial and parallel (**_pathos_** library) realization
1. **_RAM_** disk could be used for accelerate calculation (_user choice_)
1. **_numba_** library is using for a calculation speed increasing

